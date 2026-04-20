"""酒店查询工具：封装 SerpAPI 的 Google Hotels 查询并输出简化结构。"""

import os
from typing import Optional, List, Dict, Any
import serpapi
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.tools import tool

class HotelsInput(BaseModel):
    q: str = Field(description='Location of the hotel')
    check_in_date: str = Field(description='Check-in date. The format is YYYY-MM-DD')
    check_out_date: str = Field(description='Check-out date. The format is YYYY-MM-DD')
    sort_by: Optional[str] = Field('8', description='Sort by highest rating')
    adults: Optional[int] = Field(1, description='Number of adults')
    children: Optional[int] = Field(0, description='Number of children')
    rooms: Optional[int] = Field(1, description='Number of rooms')
    hotel_class: Optional[str] = Field(None, description='Hotel class, e.g., 4')
    currency: str = Field('CNY', description='Currency code')

@tool(args_schema=HotelsInput)
def hotels_finder(
    q: str,
    check_in_date: str,
    check_out_date: str,
    sort_by: Optional[str] = '8',
    adults: Optional[int] = 1,
    children: Optional[int] = 0,
    rooms: Optional[int] = 1,
    hotel_class: Optional[str] = None,
    currency: str = 'CNY'
):
    """Find hotels using the Google Hotels engine. Returns a cleaned list of hotel information with direct website links and image URLs."""
    
    # 处理中文城市名
    if not any(char.isascii() for char in q):
        q = q + ", China"
    
    search_params = {
        'api_key': os.environ.get('SERPAPI_API_KEY'),
        'engine': 'google_hotels',
        # 酒店接口当前使用英文区域参数，结果稳定性更高。
        'hl': 'en',
        'gl': 'us',
        'q': q,
        'check_in_date': check_in_date,
        'check_out_date': check_out_date,
        'currency': currency,
        'adults': adults,
        'children': children,
        'rooms': rooms,
        'sort_by': sort_by,
        'hotel_class': hotel_class
    }
    
    try:
        search = serpapi.search(search_params)
        results = search.data
        properties = results.get('properties', [])
        
        cleaned_hotels = []
        for hotel in properties[:5]:  # 只处理前5个
            # 提取酒店官网链接（如果有）
            website_link = None
            # 常见的官网链接字段
            if 'link' in hotel:
                website_link = hotel['link']
            elif 'website' in hotel:
                website_link = hotel['website']
            
            # 提取图片链接（用于展示酒店样子）
            image_link = None
            if 'images' in hotel and hotel['images']:
                image_link = hotel['images'][0].get('thumbnail', '')
            elif 'image' in hotel:
                image_link = hotel['image']
            
            cleaned_hotel = {
                'name': hotel.get('name', 'N/A'),
                'description': hotel.get('description', 'No description available'),
                'location': hotel.get('address', 'No address provided'),
                'rate_per_night': hotel.get('rate_per_night', 'N/A'),
                'total_rate': hotel.get('total_rate', 'N/A'),
                'rating': hotel.get('overall_rating', 'N/A'),
                'reviews': hotel.get('reviews', 'N/A'),
                'amenities': hotel.get('amenities', []),
                'website_link': website_link,  # 可直接点击的酒店官网
                'image_link': image_link       # 可查看酒店样子的图片链接
            }
            cleaned_hotels.append(cleaned_hotel)
        
        # 返回统一结构，便于上层 LLM 直接组织回复内容。
        return cleaned_hotels
        
    except Exception as e:
        return f"Error fetching hotels: {str(e)}"






















# import os
# from typing import Optional

# import serpapi
# from langchain.pydantic_v1 import BaseModel, Field
# from langchain_core.tools import tool

# # from pydantic import BaseModel, Field


# class HotelsInput(BaseModel):
#     q: str = Field(description='Location of the hotel')
#     check_in_date: str = Field(description='Check-in date. The format is YYYY-MM-DD. e.g. 2024-06-22')
#     check_out_date: str = Field(description='Check-out date. The format is YYYY-MM-DD. e.g. 2024-06-28')
#     sort_by: Optional[str] = Field(8, description='Parameter is used for sorting the results. Default is sort by highest rating')
#     adults: Optional[int] = Field(1, description='Number of adults. Default to 1.')
#     children: Optional[int] = Field(0, description='Number of children. Default to 0.')
#     rooms: Optional[int] = Field(1, description='Number of rooms. Default to 1.')
#     hotel_class: Optional[str] = Field(
#         None, description='Parameter defines to include only certain hotel class in the results. for example- 2,3,4')


# class HotelsInputSchema(BaseModel):
#     params: HotelsInput


# @tool(args_schema=HotelsInputSchema)
# def hotels_finder(params: HotelsInput):
#     '''
#     Find hotels using the Google Hotels engine.

#     Returns:
#         dict: Hotel search results.
#     '''

#     params = {
#         'api_key': os.environ.get('SERPAPI_API_KEY'),
#         'engine': 'google_hotels',
#         'hl': 'en',
#         'gl': 'us',
#         'q': params.q,
#         'check_in_date': params.check_in_date,
#         'check_out_date': params.check_out_date,
#         'currency': 'USD',
#         'adults': params.adults,
#         'children': params.children,
#         'rooms': params.rooms,
#         'sort_by': params.sort_by,
#         'hotel_class': params.hotel_class
#     }

#     search = serpapi.search(params)
#     results = search.data
#     return results['properties'][:5]
