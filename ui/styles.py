def load_deep_space_theme():
    return """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Orbitron:wght@600;800;900&display=swap');

        .stApp {
            background-color: #030611;
            background-image: 
                radial-gradient(1.5px 1.5px at 40px 60px, #ffffff, rgba(0,0,0,0)),
                radial-gradient(1px 1px at 150px 220px, #b0c4de, rgba(0,0,0,0)),
                radial-gradient(2px 2px at 300px 100px, #ffffff, rgba(0,0,0,0)),
                radial-gradient(1.5px 1.5px at 500px 450px, #87ceeb, rgba(0,0,0,0)),
                radial-gradient(1px 1px at 800px 150px, #ffffff, rgba(0,0,0,0)),
                radial-gradient(2px 2px at 1050px 320px, #ffffff, rgba(0,0,0,0)),
                linear-gradient(to right, rgba(3, 6, 17, 0.94), rgba(4, 8, 24, 0.88)),
                url('https://images.unsplash.com/photo-1532693322450-2cb5c511067d?auto=format&fit=crop&w=1920&q=80');
            background-size: 100% 100%, 100% 100%, 100% 100%, 100% 100%, 100% 100%, 100% 100%, 100% 100%, cover;
            background-attachment: fixed;
            background-position: center right;
            color: #d1d5db;
            font-family: 'JetBrains Mono', monospace;
        }

        h1, h2, h3, h4 {
            font-family: 'Orbitron', sans-serif !important;
            color: #f8fafc !important;
            letter-spacing: 0.08em;
            text-shadow: 0 0 12px rgba(0, 229, 255, 0.4);
        }

        .hud-panel {
            background: rgba(8, 14, 30, 0.78);
            border: 1px solid rgba(0, 229, 255, 0.25);
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.65);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border-radius: 6px;
            padding: 16px;
            margin-bottom: 14px;
        }

        .metric-card {
            background: rgba(10, 18, 38, 0.82);
            border-left: 3px solid #00e5ff;
            border-top: 1px solid rgba(0, 229, 255, 0.2);
            border-right: 1px solid rgba(0, 229, 255, 0.2);
            border-bottom: 1px solid rgba(0, 229, 255, 0.2);
            padding: 12px 16px;
            border-radius: 4px;
            text-align: center;
        }

        .metric-label {
            font-size: 11px;
            color: #94a3b8;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .metric-value {
            font-family: 'Orbitron', sans-serif;
            font-size: 24px;
            font-weight: 800;
            color: #38bdf8;
            margin-top: 4px;
        }

        .stButton>button {
            font-family: 'Orbitron', sans-serif !important;
            background: linear-gradient(135deg, #0284c7 0%, #00e5ff 100%) !important;
            color: #020617 !important;
            font-weight: 800 !important;
            border: none !important;
            border-radius: 4px !important;
            padding: 0.6rem 2rem !important;
            letter-spacing: 0.08em !important;
            box-shadow: 0 0 15px rgba(0, 229, 255, 0.35);
        }

        .stButton>button:hover {
            box-shadow: 0 0 25px rgba(0, 229, 255, 0.7);
            color: #ffffff !important;
        }
    </style>
    """