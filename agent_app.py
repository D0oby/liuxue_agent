import os
import langchain
from exa_py import Exa
from dotenv import load_dotenv
from typing import TypedDict, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

langchain.debug = True
# 配置api key
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# 1. 定义数据结构
class StudentProfile(BaseModel):
    """用于大模型结构化输出的 Schema"""
    undergrad_uni: Optional[str] = Field(default=None, description="学生的本科院校或者高中院校，例如：悉尼大学")
    gpa: Optional[float] = Field(default=None, description="学生的绩点，例如：3.8")
    budget: Optional[float] = Field(default=None, description="学生的留学预算。无论用户提供的是每月、每年还是总预算，"
            "请你统一换算为【每年】的预算，并且单位必须是【万人民币】。"
            "例如：用户说'每月1万'，这里填 12.0；用户说'总预算60万读2年'，这里填 30.0。")

class AgentState(TypedDict):
    """LangGraph 的全局状态，随着图的运行不断更新"""
    profile: StudentProfile
    user_input: str
    ai_response: str
    is_complete: bool

# 2. 初始化 LLM 与信息提取链
llm = ChatOpenAI(model="ep-20260312193546-mq459", temperature=0)

extract_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的留学规划助手。请从用户的输入中提取本科院校、GPA和预算信息。\n"
               "如果用户没有提到某项信息，请将其保持为 null。不要主观猜测。\n"
               "当前已有的信息：{current_profile}"),
    ("user", "{user_input}")
])

# 强制 LLM 输出 StudentProfile 格式的数据
extractor_chain = extract_prompt | llm.with_structured_output(StudentProfile,method="function_calling")
# 3. 定义图中的节点 (Nodes)
def extract_info_node(state: AgentState):
    """节点 1：提炼用户输入的信息并更新状态"""
    print("\n--- [Node] 正在提炼信息 ---")
    current_profile = state.get("profile", StudentProfile())
    user_input = state["user_input"]
    
    # 调用 LLM 提取新信息
    new_info = extractor_chain.invoke({
        "current_profile": current_profile.model_dump_json(),
        "user_input": user_input
    })
    
    # 合并新旧信息 (公式中的 s_{t+1} = s_t U \Delta s_t)
    updated_profile = StudentProfile(
        undergrad_uni=new_info.undergrad_uni or current_profile.undergrad_uni,
        gpa=new_info.gpa or current_profile.gpa,
        budget=new_info.budget or current_profile.budget
    )
    
    # 检查信息是否完整
    is_complete = all([
        updated_profile.undergrad_uni is not None,
        updated_profile.gpa is not None,
        updated_profile.budget is not None
    ])
    
    return {"profile": updated_profile, "is_complete": is_complete}

def ask_user_node(state: AgentState):
    """节点 2：信息不全时，生成追问话术"""
    print("--- [Node] 发现信息缺失，准备追问 ---")
    profile = state["profile"]
    missing = []
    if not profile.undergrad_uni: missing.append("本科院校")
    if not profile.gpa: missing.append("当前GPA")
    if not profile.budget: missing.append("大概的留学预算")
    
    ai_response = f"为了给您定制最准确的留学方案，我还需了解您的：{', '.join(missing)}。请问分别是多少？"
    return {"ai_response": ai_response}

# def generate_report_node(state: AgentState):
#     """节点 3：信息收齐，调用 RAG 和搜索工具生成报告"""
#     print("--- [Node] 信息已集齐，开始规划方案 ---")
#     profile = state["profile"]
    
#     # ==========================================
#     # 这里预留 RAG 和 Search 工具的调用位置
#     # ==========================================
#     rag_results = mock_rag_tool(profile)
#     search_results = mock_search_tool(profile)
    
#     ai_response = f"太好了！根据您的背景（{profile.undergrad_uni}，GPA {profile.gpa}，预算 {profile.budget}万），我已经结合数据库和最新搜索结果为您生成了5套专属方案...\n[报告生成完毕]"
#     return {"ai_response": ai_response}

def generate_report_node(state: AgentState):
    """节点 3：信息收齐，结合搜索结果由 LLM 生成正式方案"""
    print("\n--- [Node] 正在调取实时数据并生成规划方案 ---")
    
    profile = state["profile"]
    
    # 1. 调用你之前写的 Exa 搜索工具，获取实时政策和案例
    # 这个工具会返回关于该学生背景的最新网页正文
    search_results = mock_search_tool(profile)
    print(f"DEBUG -",search_results)
    
    # 2. 定义专门用于生成报告的 Prompt
    # 我们把学生档案和搜索到的实时资料都喂给大模型
    report_prompt = f"""
    你是一名资深的留学规划专家。请根据以下学生信息和实时搜索到的政策资料，为学生量身定制一份留学方案。
    
    【学生基本信息】
    - 本科院校: {profile.undergrad_uni}
    - 绩点 (GPA): {profile.gpa}
    - 预算: {profile.budget} 万人民币/年
    
    【实时搜索到的最新政策/资料】
    {search_results}
    
    【任务要求】
    1. 结合学生的 GPA 和预算，分析其申请优势与劣势。
    2. 根据搜索到的最新政策，推荐 3-5 所适合的院校或研究项目。
    3. 给出具体的申请时间轴建议。
    4. 语言要专业、客观且富有鼓励性。
    
    请直接输出最终的规划报告：
    """

    # 3. 调用大模型生成最终回复
    # 这里直接使用 llm.invoke，因为它需要输出的是一段长文本报告，而不是结构化数据
    response = llm.invoke(report_prompt)
    
    # 4. 将生成的报告存入 ai_response，返回给全局状态
    return {"ai_response": response.content}

# 4. 预留的工具空函数 (Tools)
def mock_rag_tool(profile: StudentProfile):
    """占位：检索本地学校数据库"""
    # TODO: 接入向量数据库检索逻辑
    return "mock_rag_data"

def mock_search_tool(profile: StudentProfile):
    """占位：调用搜索引擎获取最新政策"""
    # 1. 初始化 Exa 客户端
    exa_key = os.getenv("EXA_API_KEY")
    exa = Exa(api_key=exa_key)
    # 2. 动态构造搜索词
    uni_info = profile.undergrad_uni if profile.undergrad_uni else "USYD"
    query = f"{uni_info} international student research application policy 2025"
    try:
        # 3. 调用 Exa 搜索，并直接要求返回网页的纯文本内容 (contents)
        response = exa.search_and_contents(
            query,
            num_results=2,
            text = True,
        )
        if not response.results:
            return "DEBUG: 搜索成功但未找到匹配网页。"
        # 4. 将提取到的内容拼接成一段排版清晰的字符串，返回给大模型
        search_data = f"--- 实时搜索结果 ({query}) ---\n"
        for i, result in enumerate(response.results):
            # 关键：检查 result.text 是否真的存在
            content = getattr(result, 'text', '无法提取正文')
            if not content:
                content = "该网页无正文内容"
                
            search_data += f"\n[结果 {i+1}] 标题: {result.title}\n"
            search_data += f"内容摘要: {content[:800]}...\n"
            search_data += "="*20 + "\n"
        return search_data
        
    except Exception as e:
        # 如果搜索崩溃了，不要让整个 Agent 停机，而是告诉大模型搜索失败了
        return f"调用搜索引擎失败，错误信息: {str(e)}"

# 5. 定义条件边 (Conditional Edges)
def router(state: AgentState):
    """路由器：决定下一步是去提问，还是去生成报告"""
    if state["is_complete"]:
        return "generate_report"
    return "ask_user"

# 6. 构建并编译状态图 (Graph)
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("extract_info", extract_info_node)
workflow.add_node("ask_user", ask_user_node)
workflow.add_node("generate_report", generate_report_node)

# 设置入口点
workflow.set_entry_point("extract_info")

# 添加条件路由：从 extract_info 出来后，根据状态决定去哪
workflow.add_conditional_edges(
    "extract_info",
    router,
    {
        "ask_user": "ask_user",
        "generate_report": "generate_report"
    }
)

# 设置结束点
workflow.add_edge("ask_user", END)  # 提问后结束当前执行，等待用户下一次输入
workflow.add_edge("generate_report", END) # 报告生成后结束

# 编译图
app = workflow.compile()

# 7. 终端交互主循环
if __name__ == "__main__":
    print("欢迎使用留学规划 Agent！（输入 'quit' 退出）")
    
    # 初始化空状态
    current_state = {
        "profile": StudentProfile(),
        "user_input": "",
        "ai_response": "",
        "is_complete": False
    }
    
    while True:
        user_text = input("\nUser: ")
        if user_text.lower() == 'quit':
            break
            
        current_state["user_input"] = user_text
        
        # 运行图
        # LangGraph 会自动执行节点，直到遇到 END
        result = app.invoke(current_state)
        
        # 将最新的状态保存下来，用于下一轮对话
        current_state["profile"] = result["profile"]
        current_state["is_complete"] = result["is_complete"]
        
        print(f"\nAI: {result['ai_response']}")
        print(f"[当前系统后台状态: {current_state['profile']}]")