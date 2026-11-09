import asyncio
import json
import urllib.request
import websockets

async def main():
    uri = "ws://127.0.0.1:8000/ws/chat"

    # Client 1: send a request, then disconnect quickly.
    ws = await websockets.connect(uri)
    await ws.send(json.dumps({
        "query": "What is the cancellation policy?"
    }))
    await asyncio.sleep(0.1)
    await ws.close()
    print("PASS: First client disconnected.")

    # Allow the server to finish handling the disconnect.
    await asyncio.sleep(2)

    # Verify HTTP server is still healthy.
    with urllib.request.urlopen(
        "http://127.0.0.1:8000/health",
        timeout=5
    ) as response:
        health = json.loads(response.read().decode())
        assert response.status == 200
        assert health["status"] == "ok"

    print("PASS: Health endpoint works after disconnect.")

    # Client 2: prove a new WebSocket client can still chat.
    async with websockets.connect(uri) as ws2:
        await ws2.send(json.dumps({
            "query": "What is the cancellation time limit?"
        }))
        reply = json.loads(await ws2.recv())
        assert reply["type"] == "message", reply
        assert reply["response"]["answer"]

    print("PASS: A new WebSocket client received a response.")
    print("PASS: Disconnect recovery test completed.")

asyncio.run(main())
