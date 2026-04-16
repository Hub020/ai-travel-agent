import os
import serpapi
from datetime import datetime, timedelta
from langchain_core.tools import tool

@tool
def find_cheap_flight_dates(departure_id: str, arrival_id: str, start_date: str, end_date: str):
    """
    查询从 departure_id 到 arrival_id 在 start_date 到 end_date 期间内，最便宜的出发日期。
    返回日期和最低价格。
    """
    api_key = os.getenv("SERPAPI_API_KEY")
    # 这里简化：只搜索 start_date 到 end_date 之间每天的航班，找出最低价
    # 注意：Google Flights API 不直接支持批量日期范围，但可以用 'departure_date' 参数指定具体日期循环。
    # 为避免过多 API 调用，可只查询几个周末/工作日，或使用 SerpApi 的 'flight_dates' 功能（需要企业版）。
    # 作为演示，我们使用一个免费替代：直接返回一个假数据或让 LLM 根据常识推荐。
    # 实际开发中，您可以调用第三方廉价机票 API，例如 Skyscanner 或携程。
    
    # 这里提供一个简单实现：循环未来 7 天，找出最低价（示例）
    cheapest_price = float('inf')
    cheapest_date = None
    current = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        params = {
            'api_key': api_key,
            'engine': 'google_flights',
            'departure_id': departure_id,
            'arrival_id': arrival_id,
            'outbound_date': date_str,
            'currency': 'CNY'
        }
        try:
            search = serpapi.search(params)
            best_flights = search.data.get('best_flights', [])
            if best_flights:
                price = best_flights[0].get('price', 0)
                if price and price < cheapest_price:
                    cheapest_price = price
                    cheapest_date = date_str
        except Exception as e:
            print(f"Error on {date_str}: {e}")
        current += timedelta(days=1)
    
    if cheapest_date:
        return f"最便宜的出发日期是 {cheapest_date}，最低价格约为 ¥{cheapest_price}。"
    else:
        return "未找到合适航班，请尝试其他日期或目的地。"