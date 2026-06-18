import asyncio
import json
import uuid
import random
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import StreamingResponse
import redis.asyncio as aioredis

from app.config import settings

# Global shared operational memory states
redis_client: aioredis.Redis = None
request_buffer_queue: asyncio.Queue = None

class InferenceRequestPayload:
    """Encapsulates isolated tracking states across the batching boundary layers."""
    def __init__(self, prompt: str):
        self.request_id = str(uuid.uuid4())
        self.prompt = prompt
        self.token_delivery_queue = asyncio.Queue()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes high-concurrency loops and distributed system states on container boot."""
    global redis_client, request_buffer_queue
    redis_client = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
    request_buffer_queue = asyncio.Queue()
    
    # Establish initial circuit tracker metrics
    await redis_client.set("cb:state", "CLOSED")
    await redis_client.set("cb:failure_count", 0)
    
    # Spawn the background continuous optimization batcher loop
    batcher_task = asyncio.create_task(continuous_dynamic_batcher())
    yield
    # Graceful shutdown routines
    batcher_task.cancel()
    await redis_client.close()

app = FastAPI(title="Low-Latency LLM Inference Gateway", lifespan=lifespan)

async def simulate_inference_processing(batch: list, fallback_active: bool):
    """Simulates multi-request generation, passing token text sequences into individual client queues."""
    tokens_to_generate = 5
    matrix_shape = f"{len(batch)}x64"
    engine_label = "[LOCAL_FALLBACK_CPU]" if fallback_active else "[PRIMARY_GPU_vLLM]"
    
    for i in range(tokens_to_generate):
        await asyncio.sleep(0.05) # Simulated token generation delay step (50ms)
        for req in batch:
            chunk_data = {
                "token": f"token_{i} ",
                "request_id": req.request_id,
                "matrix_shape_processed": matrix_shape,
                "engine_used": engine_label
            }
            await req.token_delivery_queue.put(json.dumps(chunk_data))
            
    # Close client pipelines safely
    for req in batch:
        await req.token_delivery_queue.put(None)

async def continuous_dynamic_batcher():
    """Aggregates isolated inbound client connections into 2D tensor matrices based on window timers."""
    while True:
        batch = []
        start_time = asyncio.get_event_loop().time()
        
        while len(batch) < settings.MAX_BATCH_SIZE:
            elapsed_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            time_left_ms = settings.BATCH_TIMEOUT_MS - elapsed_ms
            
            if time_left_ms <= 0:
                break
                
            try:
                # Poll the global queue for active connections matching our sliding buffer window
                req = await asyncio.wait_for(request_buffer_queue.get(), timeout=max(0, time_left_ms / 1000.0))
                batch.append(req)
            except asyncio.TimeoutError:
                break
                
        if batch:
            # Check distributed circuit breaker telemetry from Redis
            cb_state = await redis_client.get("cb:state")
            fail_simulation = settings.PRIMARY_ENGINE_FAIL_SIMULATION
            
            if cb_state == "OPEN":
                # Instant redirect path to fallback architecture
                asyncio.create_task(simulate_inference_processing(batch, fallback_active=True))
            elif fail_simulation:
                # Track primary engine failure states
                await redis_client.incr("cb:failure_count")
                fails = int(await redis_client.get("cb:failure_count"))
                if fails >= 3:
                    await redis_client.set("cb:state", "OPEN", ex=10) # Trip lock circuit for 10 seconds
                    
                # Deliver immediate failures to flush clients toward fallback sequence checks
                for req in batch:
                    await req.token_delivery_queue.put("CIRCUIT_TRIPPED_RETRY")
                    await req.token_delivery_queue.put(None)
            else:
                # Standard Happy Path processing execution routing
                asyncio.create_task(simulate_inference_processing(batch, fallback_active=False))

@app.post("/api/v1/chat/stream")
async def stream_inference_chat(payload: dict):
    """API Ingestion layer that intercepts client queries and links them to the batching framework."""
    prompt = payload.get("prompt", "")
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt parameter is missing.")
        
    # Check if the circuit breaker is open to handle proactive client side re-routing instantly
    cb_state = await redis_client.get("cb:state")
    
    req_node = InferenceRequestPayload(prompt=prompt)
    await request_buffer_queue.put(req_node)
    
    async def sse_token_generator_bridge():
        while True:
            token_chunk = await req_node.token_delivery_queue.get()
            if token_chunk is None:
                break
                
            if token_chunk == "CIRCUIT_TRIPPED_RETRY":
                # Route the intercepted client request cleanly to the fallback engine matrix block
                fallback_chunk = {
                    "token": "[CIRCUIT_BREAKER_REDIRECT] -> Local Fallback Asset Active: ",
                    "request_id": req_node.request_id,
                    "matrix_shape_processed": "1x64",
                    "engine_used": "[LOCAL_FALLBACK_CPU]"
                }
                yield f"data: {json.dumps(fallback_chunk)}\n\n"
                continue
                
            yield f"data: {token_chunk}\n\n"
            
    return StreamingResponse(sse_token_generator_bridge(), media_type="text/event-stream")