#![forbid(unsafe_code)]

use std::{
    collections::{BTreeMap, BTreeSet},
    io::Read,
    net::{IpAddr, Ipv4Addr, Ipv6Addr, ToSocketAddrs},
    time::Duration,
};

use ntd_capsule::sha256;
use ntd_runtime::{
    ActionId, ActionOutput, ActionValue, ActionVerification, ActionVerifier, AdapterResult,
    CapabilityAdapter, CapabilityDescriptor, TypedAction,
};
use url::{Host, Url};

pub const DEFAULT_MAX_BODY_BYTES: usize = 512 * 1024;
pub const DEFAULT_MAX_TEXT_BYTES: usize = 64 * 1024;
pub const DEFAULT_MAX_REDIRECTS: u8 = 5;
pub const DEFAULT_SEARCH_ENDPOINT: &str = "https://html.duckduckgo.com/html/";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WebAdapterConfig {
    pub search_endpoint: String,
    pub user_agent: String,
    pub max_body_bytes: usize,
    pub max_text_bytes: usize,
    pub max_redirects: u8,
    pub connect_timeout: Duration,
    pub read_timeout: Duration,
    pub allow_private_networks: bool,
}

impl Default for WebAdapterConfig {
    fn default() -> Self {
        Self {
            search_endpoint: DEFAULT_SEARCH_ENDPOINT.into(),
            user_agent: "NTD97/0.1 sovereign-mobile-agent".into(),
            max_body_bytes: DEFAULT_MAX_BODY_BYTES,
            max_text_bytes: DEFAULT_MAX_TEXT_BYTES,
            max_redirects: DEFAULT_MAX_REDIRECTS,
            connect_timeout: Duration::from_secs(8),
            read_timeout: Duration::from_secs(15),
            allow_private_networks: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WebAdapterError {
    InvalidConfig,
    InvalidUrl,
    UnsupportedScheme,
    CredentialsNotAllowed,
    PrivateNetworkDenied(String),
    DnsResolutionFailed(String),
    RedirectLimit,
    MissingRedirectLocation,
    Transport(String),
    BodyTooLarge { limit: usize },
    InvalidUtf8,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct HttpResponse {
    status: u16,
    final_url: String,
    content_type: String,
    body: Vec<u8>,
}

pub struct ProductionWebAdapter {
    config: WebAdapterConfig,
    agent: ureq::Agent,
    completed: BTreeMap<u64, AdapterResult>,
}

impl ProductionWebAdapter {
    pub fn new(config: WebAdapterConfig) -> Result<Self, WebAdapterError> {
        validate_config(&config)?;
        let search_endpoint =
            Url::parse(&config.search_endpoint).map_err(|_| WebAdapterError::InvalidUrl)?;
        validate_url_shape(&search_endpoint)?;

        let agent = ureq::AgentBuilder::new()
            .timeout_connect(config.connect_timeout)
            .timeout_read(config.read_timeout)
            .timeout_write(config.read_timeout)
            .redirects(0)
            .user_agent(&config.user_agent)
            .build();

        Ok(Self {
            config,
            agent,
            completed: BTreeMap::new(),
        })
    }

    pub fn config(&self) -> &WebAdapterConfig {
        &self.config
    }

    fn execute_fetch(
        &self,
        action_id: ActionId,
        url: &str,
    ) -> Result<AdapterResult, WebAdapterError> {
        let response = self.get(url)?;
        if response.status == 429 || response.status >= 500 {
            return Ok(AdapterResult::Retryable {
                reason: format!("HTTP {} from {}", response.status, response.final_url),
                resume_token: None,
            });
        }

        let receipt = receipt_evidence(action_id, &response);
        let summary = format!(
            "web fetch HTTP {} from {}",
            response.status, response.final_url
        );
        let value = if is_text_content_type(&response.content_type) {
            let text = body_to_text(&response.body, self.config.max_text_bytes)?;
            ActionValue::Text(text)
        } else {
            ActionValue::Bytes(response.body.clone())
        };

        Ok(AdapterResult::Completed {
            output: ActionOutput {
                summary,
                value,
                evidence: receipt,
            },
            rollback_token: None,
        })
    }

    fn execute_search(
        &self,
        action_id: ActionId,
        query: &str,
        max_results: u16,
    ) -> Result<AdapterResult, WebAdapterError> {
        let mut search = Url::parse(&self.config.search_endpoint)
            .map_err(|_| WebAdapterError::InvalidUrl)?;
        search.query_pairs_mut().append_pair("q", query);

        let response = self.get(search.as_str())?;
        if response.status == 429 || response.status >= 500 {
            return Ok(AdapterResult::Retryable {
                reason: format!("HTTP {} from search endpoint", response.status),
                resume_token: None,
            });
        }

        let text = body_to_text(&response.body, self.config.max_text_bytes)?;
        let results = extract_search_links(
            &text,
            &response.final_url,
            usize::from(max_results),
        );
        let mut evidence = receipt_evidence(action_id, &response);
        evidence.push(format!("query_sha256={}", digest_hex(&sha256(query.as_bytes()))));
        evidence.push(format!("result_count={}", results.len()));

        Ok(AdapterResult::Completed {
            output: ActionOutput {
                summary: format!("web search returned {} result(s)", results.len()),
                value: ActionValue::TextList(results),
                evidence,
            },
            rollback_token: None,
        })
    }

    fn get(&self, input: &str) -> Result<HttpResponse, WebAdapterError> {
        let mut current = Url::parse(input).map_err(|_| WebAdapterError::InvalidUrl)?;

        for redirect_index in 0..=self.config.max_redirects {
            validate_parsed_url(&current, self.config.allow_private_networks)?;
            let response = match self
                .agent
                .get(current.as_str())
                .set("Accept", "text/html,text/plain,application/json,application/octet-stream;q=0.7,*/*;q=0.5")
                .call()
            {
                Ok(response) => response,
                Err(ureq::Error::Status(_, response)) => response,
                Err(error) => return Err(WebAdapterError::Transport(error.to_string())),
            };

            let status = response.status();
            if (300..400).contains(&status) {
                if redirect_index >= self.config.max_redirects {
                    return Err(WebAdapterError::RedirectLimit);
                }
                let location = response
                    .header("Location")
                    .ok_or(WebAdapterError::MissingRedirectLocation)?;
                current = current
                    .join(location)
                    .map_err(|_| WebAdapterError::InvalidUrl)?;
                continue;
            }

            let content_type = response
                .header("Content-Type")
                .unwrap_or("application/octet-stream")
                .trim()
                .to_owned();
            let body = read_bounded(response.into_reader(), self.config.max_body_bytes)?;
            return Ok(HttpResponse {
                status,
                final_url: current.to_string(),
                content_type,
                body,
            });
        }

        Err(WebAdapterError::RedirectLimit)
    }
}

impl CapabilityAdapter for ProductionWebAdapter {
    fn execute(
        &mut self,
        action_id: ActionId,
        action: &TypedAction,
    ) -> Result<AdapterResult, String> {
        if let Some(cached) = self.completed.get(&action_id.0) {
            return Ok(cached.clone());
        }

        let result = match action {
            TypedAction::WebFetch { url } => self.execute_fetch(action_id, url),
            TypedAction::WebSearch { query, max_results } => {
                self.execute_search(action_id, query, *max_results)
            }
            _ => return Err("production web adapter received a non-web action".into()),
        }
        .map_err(|error| format!("{error:?}"))?;

        if matches!(result, AdapterResult::Completed { .. }) {
            self.completed.insert(action_id.0, result.clone());
        }
        Ok(result)
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct ProductionWebVerifier;

impl ActionVerifier for ProductionWebVerifier {
    fn verify(
        &mut self,
        descriptor: &CapabilityDescriptor,
        action: &TypedAction,
        output: &ActionOutput,
    ) -> ActionVerification {
        if !matches!(action, TypedAction::WebSearch { .. } | TypedAction::WebFetch { .. }) {
            return ActionVerification::Reject {
                reason: "non-web action supplied to web verifier".into(),
            };
        }
        if descriptor.domain != ntd_runtime::CapabilityDomain::Web {
            return ActionVerification::Reject {
                reason: "web output bound to non-web capability".into(),
            };
        }

        let status = evidence_value(&output.evidence, "status")
            .and_then(|value| value.parse::<u16>().ok());
        let Some(status) = status else {
            return ActionVerification::Reject {
                reason: "missing HTTP status receipt".into(),
            };
        };
        if !(200..300).contains(&status) {
            return ActionVerification::Reject {
                reason: format!("HTTP status {status} is not successful"),
            };
        }
        let hash_ok = evidence_value(&output.evidence, "sha256")
            .is_some_and(|value| value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit()));
        if !hash_ok {
            return ActionVerification::Reject {
                reason: "missing or invalid response digest".into(),
            };
        }

        match (&output.value, action) {
            (ActionValue::TextList(items), TypedAction::WebSearch { .. }) if !items.is_empty() => {
                ActionVerification::Accept
            }
            (ActionValue::Text(text), TypedAction::WebFetch { .. }) if !text.trim().is_empty() => {
                ActionVerification::Accept
            }
            (ActionValue::Bytes(bytes), TypedAction::WebFetch { .. }) if !bytes.is_empty() => {
                ActionVerification::Accept
            }
            _ => ActionVerification::Reject {
                reason: "web action returned no usable verified content".into(),
            },
        }
    }
}

fn validate_config(config: &WebAdapterConfig) -> Result<(), WebAdapterError> {
    if config.search_endpoint.trim().is_empty()
        || config.user_agent.trim().is_empty()
        || config.max_body_bytes == 0
        || config.max_text_bytes == 0
        || config.max_text_bytes > config.max_body_bytes
        || config.max_redirects == 0
        || config.connect_timeout.is_zero()
        || config.read_timeout.is_zero()
    {
        return Err(WebAdapterError::InvalidConfig);
    }
    Ok(())
}

pub fn validate_url(input: &str, allow_private_networks: bool) -> Result<(), WebAdapterError> {
    let parsed = Url::parse(input).map_err(|_| WebAdapterError::InvalidUrl)?;
    validate_parsed_url(&parsed, allow_private_networks)
}

fn validate_url_shape(url: &Url) -> Result<(), WebAdapterError> {
    if !matches!(url.scheme(), "http" | "https") {
        return Err(WebAdapterError::UnsupportedScheme);
    }
    if !url.username().is_empty() || url.password().is_some() {
        return Err(WebAdapterError::CredentialsNotAllowed);
    }
    if url.host().is_none() {
        return Err(WebAdapterError::InvalidUrl);
    }
    Ok(())
}

fn validate_parsed_url(
    url: &Url,
    allow_private_networks: bool,
) -> Result<(), WebAdapterError> {
    validate_url_shape(url)?;

    let host = url.host().ok_or(WebAdapterError::InvalidUrl)?;
    if allow_private_networks {
        return Ok(());
    }

    match host {
        Host::Ipv4(address) => reject_non_public(IpAddr::V4(address)),
        Host::Ipv6(address) => reject_non_public(IpAddr::V6(address)),
        Host::Domain(domain) => {
            let port = url
                .port_or_known_default()
                .ok_or(WebAdapterError::InvalidUrl)?;
            let resolved = (domain, port)
                .to_socket_addrs()
                .map_err(|_| WebAdapterError::DnsResolutionFailed(domain.to_owned()))?
                .collect::<Vec<_>>();
            if resolved.is_empty() {
                return Err(WebAdapterError::DnsResolutionFailed(domain.to_owned()));
            }
            for address in resolved {
                reject_non_public(address.ip())?;
            }
            Ok(())
        }
    }
}

fn reject_non_public(address: IpAddr) -> Result<(), WebAdapterError> {
    let denied = match address {
        IpAddr::V4(address) => !is_public_ipv4(address),
        IpAddr::V6(address) => !is_public_ipv6(address),
    };
    if denied {
        Err(WebAdapterError::PrivateNetworkDenied(address.to_string()))
    } else {
        Ok(())
    }
}

fn is_public_ipv4(address: Ipv4Addr) -> bool {
    let octets = address.octets();
    if address.is_private()
        || address.is_loopback()
        || address.is_link_local()
        || address.is_unspecified()
        || address.is_multicast()
        || address.is_broadcast()
        || address.is_documentation()
    {
        return false;
    }
    if octets[0] == 0
        || octets[0] >= 240
        || (octets[0] == 100 && (64..=127).contains(&octets[1]))
        || (octets[0] == 192 && octets[1] == 0 && octets[2] == 0)
        || (octets[0] == 198 && matches!(octets[1], 18 | 19))
    {
        return false;
    }
    true
}

fn is_public_ipv6(address: Ipv6Addr) -> bool {
    let segments = address.segments();
    if address.is_loopback() || address.is_unspecified() || address.is_multicast() {
        return false;
    }
    if (segments[0] & 0xfe00) == 0xfc00
        || (segments[0] & 0xffc0) == 0xfe80
        || (segments[0] == 0x2001 && segments[1] == 0x0db8)
    {
        return false;
    }
    if let Some(v4) = address.to_ipv4_mapped() {
        return is_public_ipv4(v4);
    }
    true
}

fn read_bounded(
    mut reader: impl Read,
    max_body_bytes: usize,
) -> Result<Vec<u8>, WebAdapterError> {
    let limit = u64::try_from(max_body_bytes)
        .map_err(|_| WebAdapterError::BodyTooLarge { limit: max_body_bytes })?
        .saturating_add(1);
    let mut bytes = Vec::with_capacity(max_body_bytes.min(64 * 1024));
    reader
        .by_ref()
        .take(limit)
        .read_to_end(&mut bytes)
        .map_err(|error| WebAdapterError::Transport(error.to_string()))?;
    if bytes.len() > max_body_bytes {
        return Err(WebAdapterError::BodyTooLarge {
            limit: max_body_bytes,
        });
    }
    Ok(bytes)
}

fn body_to_text(body: &[u8], max_text_bytes: usize) -> Result<String, WebAdapterError> {
    let text = std::str::from_utf8(body).map_err(|_| WebAdapterError::InvalidUtf8)?;
    let mut plain = strip_markup(text);
    if plain.len() > max_text_bytes {
        let mut end = max_text_bytes;
        while !plain.is_char_boundary(end) {
            end = end.saturating_sub(1);
        }
        plain.truncate(end);
    }
    Ok(plain)
}

fn strip_markup(input: &str) -> String {
    let mut output = String::with_capacity(input.len());
    let mut in_tag = false;
    let mut pending_space = false;

    for ch in input.chars() {
        match ch {
            '<' => {
                in_tag = true;
                pending_space = true;
            }
            '>' if in_tag => {
                in_tag = false;
            }
            _ if in_tag => {}
            ch if ch.is_whitespace() => {
                pending_space = true;
            }
            ch => {
                if pending_space && !output.is_empty() {
                    output.push(' ');
                }
                pending_space = false;
                output.push(ch);
            }
        }
    }
    output.trim().to_owned()
}

fn extract_search_links(body: &str, base_url: &str, max_results: usize) -> Vec<String> {
    if max_results == 0 {
        return Vec::new();
    }
    let Ok(base) = Url::parse(base_url) else {
        return Vec::new();
    };
    let base_host = base.host_str().unwrap_or_default();
    let mut unique = BTreeSet::new();
    let mut results = Vec::new();

    for raw in href_values(body) {
        let Ok(candidate) = base.join(&raw) else {
            continue;
        };
        let candidate = normalize_search_redirect(candidate);
        if !matches!(candidate.scheme(), "http" | "https") {
            continue;
        }
        if candidate.host_str() == Some(base_host) {
            continue;
        }
        let rendered = candidate.to_string();
        if unique.insert(rendered.clone()) {
            results.push(rendered);
            if results.len() >= max_results {
                break;
            }
        }
    }
    results
}

fn href_values(body: &str) -> Vec<String> {
    let mut values = Vec::new();
    let mut rest = body;
    while let Some(index) = rest.find("href=") {
        rest = &rest[index + 5..];
        let Some(quote) = rest.chars().next() else {
            break;
        };
        if quote != '"' && quote != '\'' {
            continue;
        }
        rest = &rest[quote.len_utf8()..];
        let Some(end) = rest.find(quote) else {
            break;
        };
        values.push(rest[..end].to_owned());
        rest = &rest[end + quote.len_utf8()..];
    }
    values
}

fn normalize_search_redirect(url: Url) -> Url {
    if let Some(encoded) = url
        .query_pairs()
        .find_map(|(key, value)| (key == "uddg").then(|| value.into_owned()))
    {
        if let Ok(decoded) = Url::parse(&encoded) {
            return decoded;
        }
    }
    url
}

fn receipt_evidence(action_id: ActionId, response: &HttpResponse) -> Vec<String> {
    vec![
        "web.receipt.v1".into(),
        format!("action_id={}", action_id.0),
        format!("status={}", response.status),
        format!("final_url={}", response.final_url),
        format!("content_type={}", response.content_type),
        format!("bytes={}", response.body.len()),
        format!("sha256={}", digest_hex(&sha256(&response.body))),
    ]
}

fn evidence_value<'a>(evidence: &'a [String], key: &str) -> Option<&'a str> {
    let prefix = format!("{key}=");
    evidence
        .iter()
        .find_map(|item| item.strip_prefix(&prefix))
}

fn is_text_content_type(content_type: &str) -> bool {
    let content_type = content_type.to_ascii_lowercase();
    content_type.starts_with("text/")
        || content_type.contains("json")
        || content_type.contains("xml")
        || content_type.contains("javascript")
}

fn digest_hex(digest: &[u8; 32]) -> String {
    use std::fmt::Write as _;

    let mut out = String::with_capacity(64);
    for byte in digest {
        write!(&mut out, "{byte:02x}").expect("digest formatting cannot fail");
    }
    out
}

#[cfg(test)]
mod tests {
    use std::{
        io::{Read, Write},
        net::TcpListener,
        sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        },
        thread,
    };

    use ntd_core::{CapabilityId, SideEffectClass};
    use ntd_runtime::{CapabilityDomain, CapabilityDescriptor};

    use super::*;

    fn local_server(body: &'static str, requests: Arc<AtomicUsize>) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let address = listener.local_addr().expect("address");
        thread::spawn(move || {
            for stream in listener.incoming().take(2) {
                let mut stream = stream.expect("stream");
                let mut request = [0u8; 2048];
                let _ = stream.read(&mut request);
                requests.fetch_add(1, Ordering::SeqCst);
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                stream.write_all(response.as_bytes()).expect("write");
            }
        });
        format!("http://{address}/")
    }

    fn test_config(endpoint: String) -> WebAdapterConfig {
        WebAdapterConfig {
            search_endpoint: endpoint,
            allow_private_networks: true,
            ..WebAdapterConfig::default()
        }
    }

    fn descriptor(id: &str) -> CapabilityDescriptor {
        CapabilityDescriptor::new(
            CapabilityId(id.into()),
            1,
            CapabilityDomain::Web,
            SideEffectClass::ReadOnly,
        )
        .expect("descriptor")
    }

    #[test]
    fn public_policy_rejects_private_targets() {
        assert!(matches!(
            validate_url("http://127.0.0.1/private", false),
            Err(WebAdapterError::PrivateNetworkDenied(_))
        ));
        assert!(validate_url("http://127.0.0.1/private", true).is_ok());
        assert_eq!(
            validate_url("file:///etc/passwd", false),
            Err(WebAdapterError::UnsupportedScheme)
        );
    }

    #[test]
    fn fetch_returns_digest_receipt_and_reuses_action_id_result() {
        let calls = Arc::new(AtomicUsize::new(0));
        let endpoint = local_server("<html><body>Hello NTD97</body></html>", Arc::clone(&calls));
        let mut adapter =
            ProductionWebAdapter::new(test_config(endpoint.clone())).expect("adapter");
        let action = TypedAction::WebFetch { url: endpoint };

        let first = adapter.execute(ActionId(7), &action).expect("first");
        let second = adapter.execute(ActionId(7), &action).expect("cached");

        assert_eq!(first, second);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        let AdapterResult::Completed { output, .. } = first else {
            panic!("expected completed");
        };
        assert_eq!(output.value, ActionValue::Text("Hello NTD97".into()));
        assert!(output.evidence.iter().any(|item| item == "status=200"));
        assert!(output
            .evidence
            .iter()
            .any(|item| item.starts_with("sha256=")));

        let mut verifier = ProductionWebVerifier;
        assert_eq!(
            verifier.verify(&descriptor("web.fetch"), &action, &output),
            ActionVerification::Accept
        );
    }

    #[test]
    fn search_extracts_external_links_and_caps_results() {
        let calls = Arc::new(AtomicUsize::new(0));
        let endpoint = local_server(
            "<a href=\"https://example.com/a\">A</a><a href=\"https://example.org/b\">B</a>",
            calls,
        );
        let mut adapter =
            ProductionWebAdapter::new(test_config(endpoint)).expect("adapter");
        let action = TypedAction::WebSearch {
            query: "ntd97".into(),
            max_results: 1,
        };

        let result = adapter.execute(ActionId(9), &action).expect("search");
        let AdapterResult::Completed { output, .. } = result else {
            panic!("completed");
        };
        assert_eq!(
            output.value,
            ActionValue::TextList(vec!["https://example.com/a".into()])
        );
        let mut verifier = ProductionWebVerifier;
        assert_eq!(
            verifier.verify(&descriptor("web.search"), &action, &output),
            ActionVerification::Accept
        );
    }

    #[test]
    fn verifier_rejects_unsuccessful_http_receipt() {
        let action = TypedAction::WebFetch {
            url: "https://example.com/missing".into(),
        };
        let output = ActionOutput {
            summary: "missing".into(),
            value: ActionValue::Text("not found".into()),
            evidence: vec![
                "web.receipt.v1".into(),
                "status=404".into(),
                format!("sha256={}", "0".repeat(64)),
            ],
        };

        let mut verifier = ProductionWebVerifier;
        assert!(matches!(
            verifier.verify(&descriptor("web.fetch"), &action, &output),
            ActionVerification::Reject { .. }
        ));
    }
}
