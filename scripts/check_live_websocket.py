import asyncio
import json
import websockets

async def main():
    uri = "ws://127.0.0.1:8000/ws/chat"

    async with websockets.connect(uri) as ws:
        print("WebSocket connected successfully.")

        # First turn
        await ws.send(json.dumps({
            "query": "What is the cancellation policy?"
        }))
        first = json.loads(await ws.recv())
        print("\n--- TURN 1 ---")
        print(json.dumps(first, indent=2))

        # Second turn on the same connection
        await ws.send(json.dumps({
            "query": "Can you clarify the cancellation time limit?"
        }))
        second = json.loads(await ws.recv())
        print("\n--- TURN 2 ---")
        print(json.dumps(second, indent=2))

        assert first["type"] == "message", "First turn failed"
        assert second["type"] == "message", "Second turn failed"
        assert first["session_id"] == second["session_id"], \
            "Session changed between turns"

        print("\nPASS: Multi-turn chat works on one WebSocket connection.")
        print("PASS: Session ID remained stable.")

asyncio.run(main())
