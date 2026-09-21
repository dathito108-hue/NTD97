#![forbid(unsafe_code)]

use std::{env, fs, process};

use ntd_assimilation::{GgufConversionPlan, GgufModel};

fn main() {
    let mut args = env::args();
    let program = args.next().unwrap_or_else(|| "gguf-intake".into());
    let Some(path) = args.next() else {
        eprintln!("usage: {program} <model.gguf>");
        process::exit(2);
    };
    if args.next().is_some() {
        eprintln!("usage: {program} <model.gguf>");
        process::exit(2);
    }

    let bytes = match fs::read(&path) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("failed to read {path}: {error}");
            process::exit(1);
        }
    };
    let model = match GgufModel::parse(&bytes) {
        Ok(model) => model,
        Err(error) => {
            eprintln!("GGUF intake rejected: {error:?}");
            process::exit(1);
        }
    };
    let plan = match GgufConversionPlan::from_model(&model) {
        Ok(plan) => plan,
        Err(error) => {
            eprintln!("GGUF conversion planning rejected: {error:?}");
            process::exit(1);
        }
    };

    println!("architecture={}", plan.architecture);
    println!(
        "model_name={}",
        plan.model_name.as_deref().unwrap_or("<unnamed>")
    );
    println!("tokenizer={}", plan.tokenizer_model);
    println!("vocabulary_size={}", plan.vocabulary_size);
    println!("tensor_count={}", plan.tensor_count);
    println!("direct_tensor_count={}", plan.direct_tensor_count);
    println!("transcode_tensor_count={}", plan.transcode_tensor_count);
    println!("unsupported_tensor_count={}", plan.unsupported_tensor_count);
    println!("alignment={}", plan.alignment);
    println!("activation_ready={}", plan.activation_ready());
    for blocker in &plan.blockers {
        println!("blocker={blocker}");
    }

    if !plan.activation_ready() {
        process::exit(3);
    }
}
