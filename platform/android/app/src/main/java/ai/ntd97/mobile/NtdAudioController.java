package ai.ntd97.mobile;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.media.AudioAttributes;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.AudioTrack;
import android.media.MediaRecorder;

import java.util.Arrays;

public final class NtdAudioController {
    private static final int INPUT_SAMPLE_RATE = 16_000;
    private static final int OUTPUT_SAMPLE_RATE = 24_000;

    private volatile boolean captureRunning;
    private volatile boolean playbackRunning;
    private Thread captureThread;
    private Thread playbackThread;
    private AudioRecord audioRecord;
    private AudioTrack audioTrack;

    public boolean startCapture(Context context) {
        if (captureRunning) {
            return true;
        }
        if (context.checkSelfPermission(Manifest.permission.RECORD_AUDIO)
                != PackageManager.PERMISSION_GRANTED) {
            return false;
        }

        int minBytes = AudioRecord.getMinBufferSize(
                INPUT_SAMPLE_RATE,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT);
        if (minBytes <= 0) {
            return false;
        }

        try {
            audioRecord = new AudioRecord(
                    MediaRecorder.AudioSource.VOICE_RECOGNITION,
                    INPUT_SAMPLE_RATE,
                    AudioFormat.CHANNEL_IN_MONO,
                    AudioFormat.ENCODING_PCM_16BIT,
                    Math.max(minBytes, 4096));
            audioRecord.startRecording();
        } catch (RuntimeException error) {
            releaseCapture();
            return false;
        }

        captureRunning = true;
        captureThread = new Thread(this::captureLoop, "ntd97-audio-input");
        captureThread.start();
        return true;
    }

    public void stopCapture() {
        captureRunning = false;
        releaseCapture();
    }

    public boolean startPlayback() {
        if (playbackRunning) {
            return true;
        }

        int minBytes = AudioTrack.getMinBufferSize(
                OUTPUT_SAMPLE_RATE,
                AudioFormat.CHANNEL_OUT_MONO,
                AudioFormat.ENCODING_PCM_16BIT);
        if (minBytes <= 0) {
            return false;
        }

        try {
            audioTrack = new AudioTrack.Builder()
                    .setAudioAttributes(new AudioAttributes.Builder()
                            .setUsage(AudioAttributes.USAGE_ASSISTANCE_ACCESSIBILITY)
                            .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                            .build())
                    .setAudioFormat(new AudioFormat.Builder()
                            .setSampleRate(OUTPUT_SAMPLE_RATE)
                            .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                            .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                            .build())
                    .setTransferMode(AudioTrack.MODE_STREAM)
                    .setBufferSizeInBytes(Math.max(minBytes, 8192))
                    .build();
            audioTrack.play();
        } catch (RuntimeException error) {
            releasePlayback();
            return false;
        }

        playbackRunning = true;
        playbackThread = new Thread(this::playbackLoop, "ntd97-audio-output");
        playbackThread.start();
        return true;
    }

    public void stopPlayback() {
        playbackRunning = false;
        releasePlayback();
    }

    public void close() {
        stopCapture();
        stopPlayback();
    }

    private void captureLoop() {
        short[] buffer = new short[1024];
        while (captureRunning) {
            AudioRecord record = audioRecord;
            if (record == null) {
                break;
            }

            int read = record.read(buffer, 0, buffer.length, AudioRecord.READ_BLOCKING);
            if (read <= 0) {
                continue;
            }

            NtdRuntimeHost host = NtdSessionController.runtime();
            if (host != null) {
                host.acceptMicrophonePcm(Arrays.copyOf(buffer, read), INPUT_SAMPLE_RATE);
            }
        }
    }

    private void playbackLoop() {
        while (playbackRunning) {
            AudioTrack track = audioTrack;
            NtdRuntimeHost host = NtdSessionController.runtime();
            if (track == null || host == null) {
                sleepBriefly();
                continue;
            }

            short[] samples = host.pullSpeakerPcm(1024, OUTPUT_SAMPLE_RATE);
            if (samples == null || samples.length == 0) {
                sleepBriefly();
                continue;
            }

            track.write(samples, 0, samples.length, AudioTrack.WRITE_BLOCKING);
        }
    }

    private void releaseCapture() {
        AudioRecord record = audioRecord;
        audioRecord = null;
        if (record != null) {
            try {
                record.stop();
            } catch (IllegalStateException ignored) {
            }
            record.release();
        }
    }

    private void releasePlayback() {
        AudioTrack track = audioTrack;
        audioTrack = null;
        if (track != null) {
            try {
                track.stop();
            } catch (IllegalStateException ignored) {
            }
            track.release();
        }
    }

    private static void sleepBriefly() {
        try {
            Thread.sleep(20L);
        } catch (InterruptedException interrupted) {
            Thread.currentThread().interrupt();
        }
    }
}
