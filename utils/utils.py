import base64
from pathlib import Path
from matplotlib import pyplot as plt

plt.switch_backend("agg")

def write_html(audio_paths: list[Path], image_paths: list[Path], description: str):
    html = f"""
    <html>
    <head>
        <title>Audio and Mel Preview</title>
        <!-- Lightbox2 CSS -->
        <link href="https://cdnjs.cloudflare.com/ajax/libs/lightbox2/2.11.3/css/lightbox.min.css" rel="stylesheet" />
        <style>
            body {{
                font-family: Arial, sans-serif;
                background-color: #f9f9f9;
                margin: 0;
                padding: 0;
            }}
            .container {{
                /* Removed max-width to use full screen width */
                margin: 0 auto;
                padding: 20px;
            }}
            .description {{
                background-color: #fff;
                padding: 20px;
                margin-bottom: 20px;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                max-width: 1000px;
                margin-left: auto;
                margin-right: auto;
            }}
            .grid {{
                display: grid;
                grid-template-columns: repeat(2, 1fr); /* Set to 2 columns */
                grid-gap: 20px;
            }}
            .card {{
                background-color: #fff;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                padding: 20px;
                text-align: center;
            }}
            .card h3 {{
                margin-top: 0;
                text-transform: capitalize;
            }}
            audio {{
                width: 100%;
                margin: 10px 0;
            }}
            img {{
                width: 100%;
                height: auto;
                border-radius: 5px;
                cursor: pointer;
                transition: transform 0.2s;
            }}
            img:hover {{
                transform: scale(1.02);
            }}
            @media (max-width: 800px) {{
                .grid {{
                    grid-template-columns: 1fr; /* Stack cards on small screens */
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="description">
                <h2>Description</h2>
                <p>{description}</p>
            </div>
            <div class="grid">
    """

    names = ["real", "gen"]
    for row_name, audio_path, image_path in zip(names, audio_paths, image_paths):
        with open(audio_path, "rb") as f:
            audio_base64 = base64.b64encode(f.read()).decode("utf-8")

        with open(image_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode("utf-8")

        html += f"""
                <div class="card">
                    <h3>{row_name}</h3>
                    <audio controls>
                        <source src="data:audio/flac;base64,{audio_base64}" type="audio/flac">
                        Your browser does not support the audio element.
                    </audio>
                    <a href="data:image/png;base64,{image_base64}" data-lightbox="mel-spectrograms" data-title="{row_name} Mel Spectrogram">
                        <img src="data:image/png;base64,{image_base64}" alt="{row_name} Mel Spectrogram">
                    </a>
                </div>
        """

    html += """
            </div>
        </div>
        <!-- Lightbox2 JS -->
        <script src="https://cdnjs.cloudflare.com/ajax/libs/lightbox2/2.11.3/js/lightbox-plus-jquery.min.js"></script>
    </body>
    </html>
    """

    return html
