# pylint: disable = invalid-name
import os
import uuid
import time
import json
from datetime import date

import streamlit as st
from langchain_core.messages import HumanMessage

from agents.agent import Agent

# ---------- 配置 ----------
MAX_DAILY_CALLS = 50          # 每日最大查询次数
RATE_LIMIT_SECONDS = 10       # 两次查询最小间隔（秒）
USAGE_FILE = "api_usage.json" # 存储每日调用次数的文件


# ---------- 用户认证 ----------
def check_password():
    """验证访问密码，返回 True 表示已认证"""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    st.markdown("### 🔐 访问验证")
    password = st.text_input("请输入访问密码", type="password")
    if st.button("登录"):
        # 从 secrets 中读取密码（需要先在 .streamlit/secrets.toml 中设置 APP_PASSWORD）
        correct_password = st.secrets.get("APP_PASSWORD", "default123")
        if password == correct_password:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("密码错误，无权访问")
    return False


# ---------- 用量监控 ----------
def get_today_usage():
    """获取今天的 API 调用次数"""
    if not os.path.exists(USAGE_FILE):
        return 0
    try:
        with open(USAGE_FILE, "r") as f:
            data = json.load(f)
        today = str(date.today())
        return data.get(today, 0)
    except (json.JSONDecodeError, IOError):
        return 0


def increment_usage():
    """增加今天的调用次数"""
    today = str(date.today())
    data = {}
    if os.path.exists(USAGE_FILE):
        try:
            with open(USAGE_FILE, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = {}
    data[today] = data.get(today, 0) + 1
    with open(USAGE_FILE, "w") as f:
        json.dump(data, f)


# ---------- 限流 ----------
def rate_limit_check():
    """检查请求频率，返回 True 表示允许继续"""
    if "last_request_time" not in st.session_state:
        st.session_state.last_request_time = 0

    elapsed = time.time() - st.session_state.last_request_time
    if elapsed < RATE_LIMIT_SECONDS:
        wait = RATE_LIMIT_SECONDS - elapsed
        st.warning(f"请等待 {wait:.1f} 秒后再进行下一次查询")
        return False
    st.session_state.last_request_time = time.time()
    return True


# ---------- 原有功能函数 ----------
def populate_envs(sender_email, receiver_email, subject):
    os.environ['FROM_EMAIL'] = sender_email
    os.environ['TO_EMAIL'] = receiver_email
    os.environ['EMAIL_SUBJECT'] = subject


def send_email(sender_email, receiver_email, subject, thread_id):
    try:
        populate_envs(sender_email, receiver_email, subject)
        config = {'configurable': {'thread_id': thread_id}, 'recursion_limit': 200}
        st.session_state.agent.graph.invoke(None, config=config)
        st.success('Email sent successfully!')
        for key in ['travel_info', 'thread_id']:
            st.session_state.pop(key, None)
    except Exception as e:
        st.error(f'Error sending email: {e}')


def initialize_agent():
    if 'agent' not in st.session_state:
        st.session_state.agent = Agent()


def render_custom_css():
    st.markdown(
        '''
        <style>
        .main-title {
            font-size: 2.5em;
            color: #333;
            text-align: center;
            margin-bottom: 0.5em;
            font-weight: bold;
        }
        .sub-title {
            font-size: 1.2em;
            color: #333;
            text-align: left;
            margin-bottom: 0.5em;
        }
        .center-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            width: 100%;
        }
        .query-box {
            width: 80%;
            max-width: 600px;
            margin-top: 0.5em;
            margin-bottom: 1em;
        }
        .query-container {
            width: 80%;
            max-width: 600px;
            margin: 0 auto;
        }
        </style>
        ''', unsafe_allow_html=True)


def render_ui():
    st.markdown('<div class="center-container">', unsafe_allow_html=True)
    st.markdown('<div class="main-title">✈️🌍 AI Travel Agent 🏨🗺️</div>', unsafe_allow_html=True)
    st.markdown('<div class="query-container">', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Enter your travel query and get flight and hotel information:</div>', unsafe_allow_html=True)
    user_input = st.text_area(
        'Travel Query',
        height=200,
        key='query',
        placeholder='Type your travel query here...',
    )
    st.markdown('</div>', unsafe_allow_html=True)
    st.sidebar.image('images/ai-travel.png', caption='AI Travel Assistant')
    return user_input


def process_query(user_input):
    if user_input:
        try:
            # 限流和用量检查
            if not rate_limit_check():
                return
            if get_today_usage() >= MAX_DAILY_CALLS:
                st.error(f"今日免费查询次数已用完（{MAX_DAILY_CALLS}次/天），请明天再试。")
                return

            thread_id = str(uuid.uuid4())
            st.session_state.thread_id = thread_id

            messages = [HumanMessage(content=user_input)]
            config = {'configurable': {'thread_id': thread_id}, 'recursion_limit': 200}

            result = st.session_state.agent.graph.invoke({'messages': messages}, config=config)

            # 成功调用后增加计数
            increment_usage()

            st.subheader('Travel Information')
            st.write(result['messages'][-1].content)

            st.session_state.travel_info = result['messages'][-1].content

        except Exception as e:
            st.error(f'Error: {e}')
    else:
        st.error('Please enter a travel query.')


def render_email_form():
    send_email_option = st.radio('Do you want to send this information via email?', ('No', 'Yes'))
    if send_email_option == 'Yes':
        with st.form(key='email_form'):
            sender_email = st.text_input('Sender Email')
            receiver_email = st.text_input('Receiver Email')
            subject = st.text_input('Email Subject', 'Travel Information')
            submit_button = st.form_submit_button(label='Send Email')

        if submit_button:
            if sender_email and receiver_email and subject:
                send_email(sender_email, receiver_email, subject, st.session_state.thread_id)
            else:
                st.error('Please fill out all email fields.')


def main():
    # 先进行密码验证
    if not check_password():
        st.stop()

    initialize_agent()
    render_custom_css()

    # 侧边栏显示剩余次数
    remaining = MAX_DAILY_CALLS - get_today_usage()
    st.sidebar.metric("今日剩余查询次数", remaining)

    user_input = render_ui()

    if st.button('Get Travel Information'):
        process_query(user_input)

    if 'travel_info' in st.session_state:
        render_email_form()


if __name__ == '__main__':
    main()