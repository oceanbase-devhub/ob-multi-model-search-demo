from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional
from obmms import AgentStat, StatelessAgentFlow

app = FastAPI()

class Request(BaseModel):
    messages: List[dict] = Field(description="前端用户和机器人的对话历史")
    new_input: str = Field(description="最新用户输入")
    # stat: int = Field(description="前端agent状态机状态码")
    departure: Optional[str] = Field(description="前端保存的旅行出发地")
    distance: Optional[str] = Field(description="前端保存的行程范围")
    score: Optional[int] = Field(description="前端保存的景点评分")
    season: Optional[str] = Field(description="前端保存的季节要求")

@app.post("/obmms/chat")
async def chat(req: Request):
    agent_flow = StatelessAgentFlow(
        chat_history=req.messages,
        departure=req.departure,
        distance=req.distance,
        score=req.score,
        season=req.season,
    )
    streamer, chat_resp = await agent_flow.chat(req.new_input)

    async def do_stream():
        yield f"meta:{chat_resp.model_dump_json()}\n\n"
        if streamer is not None:
            for res in streamer:
                # msg.append(res.output.choices[0].message.content)
                yield f"data:{res.output.choices[0].message.content}\n\n"
        
    return StreamingResponse(do_stream(), media_type="text/event-stream")