# ✈️🧳 AI Travel Muti-Agent - Powered by LangGraph: A Practical Use Case 🌍
Welcome to the AI Travel Agent repository! This project demonstrates how to leverage LangGraph for building a smart travel assistant that uses multiple language models (LLMs) to handle tasks such as finding flights, booking hotels, and sending personalized emails. The agent is designed to interact with users, invoke necessary tools, and provide a seamless travel planning experience.

## **Features**

- **Stateful Interactions**: The agent remembers user interactions and continues from where it left off, ensuring a smooth user experience.
- **Human-in-the-Loop**: Users have control over critical actions, like reviewing travel plans before emails are sent.
- **Dynamic LLM Usage**: The agent intelligently switches between different LLMs for various tasks, like tool invocation and email generation.
- **Email Automation**: Automatically generates and sends detailed travel plans to users via email.

## Getting Started
Clone the repository, set up the virtual environment, and install the required packages

1. git clone git@github.com:nirbar1985/ai-travel-agent.git

1. ( In case you have python version 3.11.9 installed in pyenv)
   ```shell script
   pyenv local 3.11.9
   ```

1. Install dependencies
    ```shell script
    poetry install --sync
    ```

1. Enter virtual env by:
    ```shell script
    poetry shell
    ```

## **Store Your API Keys**

1. Create a `.env` file in the root directory of the project.
2. Add your API keys and environment variables to the `.env` file:
    ```plaintext
    OPENAI_API_KEY=your_openai_api_key
    SERPAPI_API_KEY=your_serpapi_api_key
    SENDGRID_API_KEY=your_sendgrid_api_key

    # Observability variables
    LANGCHAIN_API_KEY=your_langchain_api_key
    LANGCHAIN_TRACING_V2=true
    LANGCHAIN_PROJECT=ai_travel_agent
    ```

Make sure to replace the placeholders (`your_openai_api_key`, `your_serpapi_api_key`, `your_langchain_api_key`, `your_sendgrid_api_key`) with your actual keys.
This version includes the necessary environment variables for OpenAI, SERPAPI, LangChain, and SendGrid and the LANGCHAIN_TRACING_V2 and LANGCHAIN_PROJECT configurations.

### How to Run the Chatbot
To start the chatbot, run the following command:
```
streamlit run app.py
```

### Using the Chatbot
Once launched, simply enter your travel request. For example:
> I want to travel to Amsterdam from Madrid from October 1st to 7th. Find me flights and 4-star hotels.
我想从北京去上海，10月1日到10月7日，帮我找航班和酒店
我5月要从北京去上海，希望在本月内找到机票酒店比较实惠的日期。帮我规划轻松休闲的行程，推荐不辣的正宗美食，并告知天气和穿衣建议。



The chatbot will generate results that include logos and links for easy navigation.

> **Note**: The data is fetched via Google Flights and Google Hotels APIs. There’s no affiliation or promotion of any particular brand.


#### Example Outputs

- Flight and hotel options with relevant logos and links for easy reference:
- <img width="1595" height="897" alt="image" src="https://github.com/user-attachments/assets/4c760791-6745-42ff-928d-61353aeac45d" />


<img width="1578" height="850" alt="image" src="https://github.com/user-attachments/assets/19ec4133-8781-4ab0-8272-bd48c6878aad" />
<img width="1600" height="855" alt="image" src="https://github.com/user-attachments/assets/60e620f3-03fe-4dff-824f-7066e4e9c60c" />

<img width="1195" height="551" alt="image" src="https://github.com/user-attachments/assets/c7ffce13-e90a-4e53-a4ef-98f68d5fd63d" />

<img width="1145" height="433" alt="image" src="https://github.com/user-attachments/assets/c13ef5e9-8ae1-4dd9-9165-2f9336840edf" />


<img width="824" height="764" alt="image" src="https://github.com/user-attachments/assets/2b1cbc66-fc0a-4748-a6f2-2c87e5547a0e" />
<img width="956" height="723" alt="image" src="https://github.com/user-attachments/assets/973520ec-2204-4078-92b7-88806d3be2e4" />
<img width="853" height="352" alt="image" src="https://github.com/user-attachments/assets/b0961d69-1afb-4297-9f35-3aeb771261e6" />


#### Email Integration
The email integration is implemented using the **human-in-the-loop** feature, allowing you to stop the agent execution and return control back to the user, providing flexibility in managing the travel data before sending it via email.

![photo4](https://github.com/user-attachments/assets/53775c87-7881-40c3-9b23-2885ed020e46)

- Travel data formatted in HTML, delivered straight to your inbox:
![photo5](https://github.com/user-attachments/assets/02641ce1-b303-4020-9849-7d77f596a6ba)
![photo6](https://github.com/user-attachments/assets/1c3d8a35-148d-4144-829a-b1db6e3b3dde)

## Learn More
For a detailed explanation of the underlying technology, check out the full article on Medium:
[Building Production-Ready AI Agents with LangGraph: A Real-Life Use Case](https://medium.com/cyberark-engineering/building-production-ready-ai-agents-with-langgraph-a-real-life-use-case-7bda34c7f4e4))

## License
Distributed under the MIT License. See LICENSE.txt for more information.
