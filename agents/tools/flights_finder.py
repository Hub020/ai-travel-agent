"""航班查询工具：封装 SerpAPI 的 Google Flights 查询并做结果清洗。"""

import os
from typing import Optional
import serpapi
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.tools import tool

# 城市到机场代码映射（可继续扩充）
CITY_TO_AIRPORT = {
    "北京": "PEK", "上海": "SHA", "广州": "CAN", "深圳": "SZX",
    "成都": "CTU", "杭州": "HGH", "西安": "XIY", "重庆": "CKG",
    "南京": "NKG", "武汉": "WUH", "厦门": "XMN", "青岛": "TAO",
    "长沙": "CSX", "昆明": "KMG", "郑州": "CGO", "沈阳": "SHE",
    "大连": "DLC", "哈尔滨": "HRB", "天津": "TSN", "三亚": "SYX",
    "纽约": "JFK", "伦敦": "LHR", "东京": "NRT", "巴黎": "CDG",
    "香港": "HKG",
}

class FlightsInput(BaseModel):
    departure_id: str = Field(description='Departure airport code or location keyword')
    arrival_id: str = Field(description='Arrival airport code or location keyword')
    outbound_date: str = Field(description='Outbound date. The format is YYYY-MM-DD')
    return_date: Optional[str] = Field(None, description='Return date for round trip. The format is YYYY-MM-DD')
    currency: str = Field('CNY', description='Currency code (CNY for Chinese Yuan)')
    adults: int = Field(1, description='Number of adults')

@tool(args_schema=FlightsInput)
def flights_finder(
    departure_id: str,
    arrival_id: str,
    outbound_date: str,
    return_date: Optional[str] = None,
    currency: str = 'CNY',
    adults: int = 1
):
    """Find flights using the Google Flights engine."""
    # 兼容中文城市输入：先尝试映射到 IATA 机场三字码。
    departure_code = CITY_TO_AIRPORT.get(departure_id, departure_id)
    arrival_code = CITY_TO_AIRPORT.get(arrival_id, arrival_id)
    
    search_params = {
        'api_key': os.environ.get('SERPAPI_API_KEY'),
        'engine': 'google_flights',
        'departure_id': departure_code,
        'arrival_id': arrival_code,
        'outbound_date': outbound_date,
        'currency': currency,
        'adults': adults,
        'hl': 'zh-CN',
        'gl': 'cn',
    }
    
    if return_date:
        search_params['return_date'] = return_date
        search_params['type'] = '1'   # 往返
    else:
        search_params['type'] = '2'   # 单程
    
    search = serpapi.search(search_params)
    results = search.data
    flights = results.get('best_flights', [])[:5]
    
    # 仅保留前 5 条，并提取上层提示词所需的关键字段（尤其是 booking_link）。
    enriched_flights = []
    for flight in flights:
        enriched = {
            "airline": flight.get('airline', 'N/A'),
            "flight_number": flight.get('flight_number', 'N/A'),
            "departure_airport": flight.get('departure_airport', {}).get('name', 'N/A'),
            "departure_time": flight.get('departure_time', 'N/A'),
            "arrival_airport": flight.get('arrival_airport', {}).get('name', 'N/A'),
            "arrival_time": flight.get('arrival_time', 'N/A'),
            "duration": flight.get('duration', 'N/A'),
            "price": flight.get('price', 'N/A'),
            "booking_link": flight.get('book_on_google_flights_link', '')  # 关键
        }
        enriched_flights.append(enriched)
    # 兼容当前上层调用：返回（清洗结果, 原始前5条）二元组。
    return enriched_flights, results.get('best_flights', [])[:5]
    # return results.get('best_flights', [])[:5]


















# import os
# from typing import Optional

# # from pydantic import BaseModel, Field
# import serpapi
# from langchain.pydantic_v1 import BaseModel, Field
# from langchain_core.tools import tool


# class FlightsInput(BaseModel):
#     departure_airport: Optional[str] = Field(description='Departure airport code (IATA)')
#     arrival_airport: Optional[str] = Field(description='Arrival airport code (IATA)')
#     outbound_date: Optional[str] = Field(description='Parameter defines the outbound date. The format is YYYY-MM-DD. e.g. 2024-06-22')
#     return_date: Optional[str] = Field(description='Parameter defines the return date. The format is YYYY-MM-DD. e.g. 2024-06-28')
#     adults: Optional[int] = Field(1, description='Parameter defines the number of adults. Default to 1.')
#     children: Optional[int] = Field(0, description='Parameter defines the number of children. Default to 0.')
#     infants_in_seat: Optional[int] = Field(0, description='Parameter defines the number of infants in seat. Default to 0.')
#     infants_on_lap: Optional[int] = Field(0, description='Parameter defines the number of infants on lap. Default to 0.')


# class FlightsInputSchema(BaseModel):
#     params: FlightsInput


# @tool(args_schema=FlightsInputSchema)
# def flights_finder(params: FlightsInput):
#     '''
#     Find flights using the Google Flights engine.

#     Returns:
#         dict: Flight search results.
#     '''

#     params = {
#         'api_key': os.environ.get('SERPAPI_API_KEY'),
#         'engine': 'google_flights',
#         'hl': 'en',
#         'gl': 'us',
#         'departure_id': params.departure_airport,
#         'arrival_id': params.arrival_airport,
#         'outbound_date': params.outbound_date,
#         'return_date': params.return_date,
#         'currency': 'USD',
#         'adults': params.adults,
#         'infants_in_seat': params.infants_in_seat,
#         'stops': '1',
#         'infants_on_lap': params.infants_on_lap,
#         'children': params.children
#     }

#     try:
#         search = serpapi.search(params)
#         results = search.data['best_flights']
#     except Exception as e:
#         results = str(e)
#     return results
