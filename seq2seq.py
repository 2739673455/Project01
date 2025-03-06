import random
from datetime import datetime, timedelta
import pandas as pd

# 月份名称列表
MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
MONTHS_SHORT = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# 中文数字映射
CHINESE_NUMS = {0: "零", 1: "一", 2: "二", 3: "三", 4: "四", 5: "五", 6: "六", 7: "七", 8: "八", 9: "九"}


def number_to_chinese(num):
    """将数字转换为中文"""
    if num < 10:
        return CHINESE_NUMS[num]
    elif num < 20:
        return "十" + (CHINESE_NUMS[num % 10] if num % 10 != 0 else "")
    else:
        tens = num // 10
        ones = num % 10
        return CHINESE_NUMS[tens] + "十" + (CHINESE_NUMS[ones] if ones != 0 else "")


def year_to_chinese(year):
    """将年份转换为中文"""
    return "".join(CHINESE_NUMS[int(d)] for d in str(year))


def random_date(start_year=1900, end_year=2025):
    """生成随机日期"""
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)
    delta = end_date - start_date
    random_days = random.randint(0, delta.days)
    return start_date + timedelta(days=random_days)


def generate_date_formats(date):
    """为给定日期生成多种格式，包括中文格式"""
    year = date.year
    month = date.month
    day = date.day

    # 目标格式
    target = date.strftime("%Y-%m-%d")

    # 英文格式
    formats = [
        f"{day}/{month}/{year}",  # dd/mm/yyyy
        f"{month}/{day}/{year}",  # mm/dd/yyyy
        f"{day}-{month}-{year}",  # dd-mm-yyyy
        f"{month}-{day}-{year}",  # mm-dd-yyyy
        f"{year}/{month}/{day}",  # yyyy/mm/dd
        f"{year}-{month}-{day}",  # yyyy-mm-dd
        f"{day}.{month}.{year}",  # dd.mm.yyyy
        f"{MONTHS[month-1]} {day}, {year}",  # Month dd, yyyy
        f"{day} {MONTHS[month-1]} {year}",  # dd Month yyyy
        f"{MONTHS_SHORT[month-1]} {day}, {year}",  # Mon dd, yyyy
        f"{day:02d}/{month:02d}/{year}",  # dd/mm/yyyy with leading zeros
        f"{month:02d}/{day:02d}/{year}",  # mm/dd/yyyy with leading zeros
        f"{year}{month:02d}{day:02d}",  # yyyymmdd
        f"{day:02d}-{MONTHS_SHORT[month-1]}-{year}",  # dd-Mon-yyyy
    ]

    # 添加中文格式
    chinese_month = number_to_chinese(month)
    chinese_day = number_to_chinese(day)
    chinese_year = year_to_chinese(year)

    chinese_formats = [
        f"{chinese_year}年{month}月{day}日",  # 1990年1月3日
        f"{chinese_year}年{chinese_month}月{chinese_day}日",  # 一九九零年一月三日
        f"{year}年{month}月{day}日",  # 1990年1月3日 (混合格式)
        f"{year}年{chinese_month}月{chinese_day}日",  # 1990年一月三日 (混合格式)
        f"{chinese_year}年{month:02d}月{day:02d}日",  # 一九九零年01月03日
    ]

    formats.extend(chinese_formats)
    return [(fmt, target) for fmt in formats]


def generate_training_data(num_samples, filename="data/date.csv"):
    """生成训练数据并保存到CSV"""
    data = []

    for _ in range(num_samples):
        date = random_date()
        date_pairs = generate_date_formats(date)
        data.extend(date_pairs)

    df = pd.DataFrame(data, columns=["source", "target"])
    df.to_csv(filename, index=False)
    print(f"Generated {len(data)} samples and saved to {filename}")


if __name__ == "__main__":
    generate_training_data(10000)
