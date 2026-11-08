"""Gradio interface for the Real vs AI-generated image classifier."""

from __future__ import annotations

import gradio as gr

if __package__:
    from .inference import AI_THRESHOLD, Prediction, predict_uploaded_file
else:
    from inference import AI_THRESHOLD, Prediction, predict_uploaded_file


def run_prediction(image_path: str | None) -> tuple[str, str]:
    """Format a prediction or a friendly validation message for the UI."""
    try:
        result: Prediction = predict_uploaded_file(image_path)
    except (TypeError, ValueError) as error:
        return f"⚠️ {error}", ""

    probability_percent = result.probability_ai_generated * 100
    confidence_percent = result.confidence * 100
    details = (
        f"**Confidence kelas terpilih:** {confidence_percent:.2f}%  \n"
        f"**Probabilitas AI-generated:** {probability_percent:.2f}%  \n"
        f"Threshold AI-generated: `{AI_THRESHOLD:.2f}`"
    )
    return f"## Hasil: {result.label}", details


with gr.Blocks(title="Real vs AI-Generated Image Classifier") as demo:
    gr.Markdown(
        """
        # Klasifikasi Gambar Real vs AI-generated

        Unggah satu gambar, lalu tekan **Jalankan prediksi**. Model akan
        mengklasifikasikan gambar sebagai **Real** atau **AI-generated**.

        Aplikasi menggunakan Custom CNN v3. Confidence adalah skor model,
        bukan jaminan kebenaran atau ukuran ketidakpastian yang terkalibrasi.
        Hasil dapat keliru, khususnya pada gambar yang berbeda dari data
        pelatihan, dan model tidak mengidentifikasi generator gambar.
        """
    )

    with gr.Row():
        image_input = gr.Image(
            type="filepath",
            sources=["upload"],
            label="Pilih gambar",
            height=360,
        )
        with gr.Column():
            result_output = gr.Markdown(
                label="Klasifikasi",
                value="Hasil prediksi akan tampil di sini.",
            )
            confidence_output = gr.Markdown(label="Confidence score")

    with gr.Row():
        predict_button = gr.Button("Jalankan prediksi", variant="primary")
        clear_button = gr.ClearButton(
            [image_input, result_output, confidence_output],
            value="Bersihkan",
        )

    predict_button.click(
        fn=run_prediction,
        inputs=image_input,
        outputs=[result_output, confidence_output],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
    )
