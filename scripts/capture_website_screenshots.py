import asyncio
import base64
import json
import os
import shutil
import subprocess
import time
import urllib.request
import websockets
from pathlib import Path


def find_chrome() -> str:
    """Locate a Chrome/Edge executable on macOS, Windows or Linux."""
    candidates = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        os.path.expandvars(r"%ProgramFiles%\Google\Chrome\Application\chrome.exe"),
        os.path.expandvars(r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"),
        os.path.expandvars(r"%LocalAppData%\Google\Chrome\Application\chrome.exe"),
        os.path.expandvars(r"%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe"),
        shutil.which("google-chrome") or "",
        shutil.which("chromium") or "",
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    raise FileNotFoundError("Google Chrome or Microsoft Edge was not found on this computer.")

async def capture_all():
    out_dir = Path("results/output_photos")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Launching Headless Chrome on port 9222...")
    chrome_proc = subprocess.Popen([
        find_chrome(),
        "--headless=new",
        "--remote-debugging-port=9222",
        "--window-size=1600,1200",
        "http://localhost:8501"
    ])
    
    try:
        await asyncio.sleep(4)
        with urllib.request.urlopen("http://localhost:9222/json") as resp:
            pages = json.loads(resp.read().decode())
        target = [p for p in pages if p.get("type") == "page"][0]
        ws_url = target["webSocketDebuggerUrl"]
        
        async with websockets.connect(ws_url, max_size=50*1024*1024) as ws:
            msg_id = 0
            async def send_cmd(method, params=None):
                nonlocal msg_id
                msg_id += 1
                payload = {"id": msg_id, "method": method}
                if params:
                    payload["params"] = params
                await ws.send(json.dumps(payload))
                while True:
                    res = json.loads(await ws.recv())
                    if res.get("id") == msg_id:
                        return res

            await send_cmd("Page.enable")
            await send_cmd("Runtime.enable")
            await asyncio.sleep(2)
            
            async def select_radio_by_text(keyword):
                js = f"""
                (() => {{
                    const labels = Array.from(document.querySelectorAll('#nav a'));
                    const found = labels.find(l => l.innerText.includes('{keyword}'));
                    if (found) {{
                        found.click();
                        return found.innerText;
                    }}
                    return null;
                }})()
                """
                r = await send_cmd("Runtime.evaluate", {"expression": js, "returnByValue": True})
                val = r.get("result", {}).get("result", {}).get("value")
                print(f"Clicked radio matching '{keyword}':", val)
                return val

            async def take_screenshot(filename):
                # Small delay to ensure any chart animations finish
                await asyncio.sleep(2)
                res = await send_cmd("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": True})
                img_data = base64.b64decode(res["result"]["data"])
                target_file = out_dir / filename
                target_file.write_bytes(img_data)
                print(f"--> Saved {target_file.name} ({len(img_data):,} bytes)")

            # 1. Dashboard
            print("\nCapturing Dashboard...")
            await select_radio_by_text("Dashboard")
            await asyncio.sleep(2)
            await take_screenshot("06_website_dashboard_overview.png")

            # 2. EEG Upload & Data Manager
            print("\nCapturing EEG Upload & Data Manager...")
            await select_radio_by_text("EEG Upload")
            await asyncio.sleep(3)
            await take_screenshot("07_website_eeg_data_manager.png")

            # 3. Multichannel Signal Explorer
            print("\nCapturing Signal Explorer...")
            await select_radio_by_text("Signal Explorer")
            await asyncio.sleep(4)
            await take_screenshot("08_website_signal_explorer_multichannel_psd.png")

            # 4. Model Training & Cross-Validation
            print("\nCapturing Model Training...")
            await select_radio_by_text("Model Training")
            await asyncio.sleep(3)
            await take_screenshot("09_website_model_training_cross_validation.png")

            # 5. Seizure Detection & Prediction (Execute analysis first)
            print("\nCapturing Seizure Detection & Prediction...")
            await select_radio_by_text("Seizure Detection")
            await asyncio.sleep(2)
            
            # Click Execute Seizure Analysis
            click_exec_js = """
            (() => {
                const btns = Array.from(document.querySelectorAll('button')).filter(b => b.innerText.includes('Execute Seizure Analysis'));
                if (btns.length > 0) {
                    btns[0].click();
                    return 'clicked Execute Seizure Analysis';
                }
                return 'not found';
            })()
            """
            r_btn = await send_cmd("Runtime.evaluate", {"expression": click_exec_js, "returnByValue": True})
            print("Execute button:", r_btn.get("result", {}).get("result", {}).get("value"))
            
            # Wait for calculation to finish (check for table or timeline)
            for _ in range(25):
                await asyncio.sleep(1)
                chk_js = "document.querySelectorAll('.card.metric').length > 0"
                chk_res = await send_cmd("Runtime.evaluate", {"expression": chk_js, "returnByValue": True})
                if chk_res.get("result", {}).get("result", {}).get("value"):
                    print("Seizure calculation finished, metrics loaded!")
                    break
            await asyncio.sleep(2)
            await take_screenshot("10_website_seizure_detection_risk_timeline.png")

            # 6. Model Evaluation & Metrics
            print("\nCapturing Model Evaluation & Metrics...")
            await select_radio_by_text("Model Evaluation")
            await asyncio.sleep(3)
            await take_screenshot("11_website_model_evaluation_metrics_confusion_matrix.png")

            # 7. Documentation & Methodology
            print("\nCapturing Documentation...")
            await select_radio_by_text("Documentation")
            await asyncio.sleep(2)
            await take_screenshot("12_website_documentation_methodology.png")

    finally:
        chrome_proc.terminate()
        print("\nAll website screenshots captured successfully!")

if __name__ == "__main__":
    asyncio.run(capture_all())
