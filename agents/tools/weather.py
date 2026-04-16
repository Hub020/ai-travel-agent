import requests
import os
from langchain_core.tools import tool

@tool
def get_weather_forecast(city: str, date: str):
    """
    查询指定城市在指定日期的天气预报（仅支持未来7天）。
    返回天气状况、温度、建议衣物。
    """
    api_key = os.getenv("AMAP_API_KEY")
    # 先获取城市adcode
    geo_url = "https://restapi.amap.com/v3/geocode/geo"
    geo_params = {"address": city, "key": api_key}
    geo_resp = requests.get(geo_url, params=geo_params)
    geo_data = geo_resp.json()
    if geo_data['status'] == '1' and geo_data['geocodes']:
        adcode = geo_data['geocodes'][0]['adcode']
    else:
        return "无法获取城市代码"
    
    # 查询天气（高德只支持未来7天，需要循环日期）
    weather_url = "https://restapi.amap.com/v3/weather/weatherInfo"
    # 注意：高德未来天气需要指定 'extensions=all'，返回3天预报。更长时间需要其他API。
    # 简化：使用OpenWeatherMap或心知天气。这里用高德示例：
    params = {"city": adcode, "key": api_key, "extensions": "all"}
    resp = requests.get(weather_url, params=params)
    data = resp.json()
    if data['status'] == '1':
        forecasts = data['forecasts'][0]['casts']
        for day in forecasts:
            if day['date'] == date:
                weather = day['dayweather']
                temp = day['daytemp']
                return f"{date} 天气：{weather}，温度 {temp}°C。建议穿着：{suggest_clothing(weather, temp)}"
        return f"未找到 {date} 的天气信息，请选择未来7天内的日期。"
    else:
        return "天气查询失败"

def suggest_clothing(weather, temp):
    temp_int = int(temp)
    if "雨" in weather:
        return "带雨具，穿防水外套。"
    elif temp_int < 10:
        return "穿羽绒服、厚毛衣。"
    elif temp_int < 20:
        return "穿风衣或薄外套。"
    else:
        return "穿短袖、裙子，注意防晒。"