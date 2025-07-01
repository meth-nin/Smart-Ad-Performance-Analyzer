import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

np.random.seed(42)
random.seed(42)


def generate_ad_dataset(n_samples=1000):

    # ad text templates for different industries
    ad_templates = {
        'tech': [
            "🚀 Revolutionary AI software that transforms your business operations!",
            "💻 Cut costs by 40% with our automated solutions. Try free trial!",
            "⚡ Speed up your workflow with cutting-edge technology",
            "🔥 Join 10,000+ companies using our AI-powered platform",
            "🎯 Smart automation for modern businesses. Get started today!",
            "💡 Innovation meets efficiency. Discover our tech solutions",
            "🌟 Transform your business with AI. Limited time offer!",
            "⭐ Award-winning software trusted by industry leaders"
        ],
        'finance': [
            "💰 Secure your financial future with our investment platform",
            "📈 Grow your wealth with expert financial advice. Start now!",
            "🏦 Bank with confidence. Zero fees, maximum returns",
            "💳 Get approved for loans in 24 hours. Apply today!",
            "📊 Smart investment tools for modern investors",
            "💸 Save more, spend wisely. Download our app!",
            "🎯 Financial planning made simple. Book consultation",
            "🔒 Protect your assets with our secure solutions"
        ],
        'ecommerce': [
            "🛍️ Flash sale! 50% off on all items. Shop now!",
            "📦 Free shipping on orders over $50. Limited time!",
            "👗 New collection just dropped. Be the first to shop!",
            "🎁 Perfect gifts for every occasion. Browse our store",
            "⚡ Lightning deals! Save big on trending products",
            "🌟 Premium quality at unbeatable prices",
            "🔥 Hot deals alert! Don't miss out on savings",
            "💎 Luxury items at affordable prices. Shop today!"
        ],
        'healthcare': [
            "🏥 Book your health checkup online. Easy scheduling!",
            "💊 Prescription delivery to your door. Order now!",
            "👨‍⚕️ Consult top doctors from home. Telemedicine made simple",
            "🩺 Advanced healthcare solutions for better living",
            "💚 Your health, our priority. Comprehensive care plans",
            "🔬 Cutting-edge medical technology at your service",
            "⚕️ Trusted healthcare providers in your area",
            "💪 Stay healthy, stay happy. Wellness programs available"
        ]
    }

    platforms = ['Facebook', 'Google Ads', 'Instagram', 'LinkedIn', 'Twitter']
    industries = list(ad_templates.keys())

    # generate data
    data = []

    for i in range(n_samples):
        industry = random.choice(industries)
        ad_text = random.choice(ad_templates[industry])

        platform = np.random.choice(platforms, p=[0.3, 0.25, 0.25, 0.15, 0.05])

        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 12, 31)
        random_date = start_date + timedelta(
            seconds=random.randint(0, int((end_date - start_date).total_seconds()))
        )

        day_of_week = random_date.strftime('%A')
        hour = random.randint(0, 23)

        sentiment_keywords = {
            'positive': ['amazing', 'best', 'free', 'save', 'new', 'premium', '🚀', '⚡', '🔥', '🌟'],
            'urgent': ['limited', 'flash', 'now', 'today', 'hurry', 'don\'t miss'],
            'professional': ['trusted', 'expert', 'professional', 'secure', 'certified']
        }

        text_lower = ad_text.lower()
        sentiment_score = 0.5

        for word in sentiment_keywords['positive']:
            if word in text_lower:
                sentiment_score += 0.15
        for word in sentiment_keywords['urgent']:
            if word in text_lower:
                sentiment_score += 0.1
        for word in sentiment_keywords['professional']:
            if word in text_lower:
                sentiment_score += 0.05

        sentiment_score = min(sentiment_score, 1.0)

        base_ctr = {
            'Facebook': 0.015,
            'Google Ads': 0.035,
            'Instagram': 0.012,
            'LinkedIn': 0.008,
            'Twitter': 0.006
        }[platform]

        # CTR
        ctr = base_ctr * (0.5 + sentiment_score) * np.random.normal(1, 0.3)
        ctr = max(0.001, min(ctr, 0.15))  # realistic bounds

        impressions = np.random.randint(1000, 50000)

        # Clicks based on CTR
        clicks = int(impressions * ctr)

        base_cpc = {
            'Facebook': 1.2,
            'Google Ads': 2.5,
            'Instagram': 1.0,
            'LinkedIn': 3.5,
            'Twitter': 0.8
        }[platform]

        cpc = base_cpc * np.random.normal(1, 0.4)
        cpc = max(0.1, cpc)

        # cost
        cost = clicks * cpc

        base_cvr = {
            'tech': 0.025,
            'finance': 0.015,
            'ecommerce': 0.035,
            'healthcare': 0.020
        }[industry]

        cvr = base_cvr * (0.5 + sentiment_score) * np.random.normal(1, 0.5)
        cvr = max(0.001, min(cvr, 0.1))

        conversions = int(clicks * cvr)

        avg_order_value = {
            'tech': 250,
            'finance': 150,
            'ecommerce': 80,
            'healthcare': 120
        }[industry]

        revenue = conversions * avg_order_value * np.random.normal(1, 0.3)
        revenue = max(0, revenue)

        roi = (revenue - cost) / cost if cost > 0 else 0
        if roi > 0.5:
            performance = 'High'
        elif roi > 0:
            performance = 'Medium'
        else:
            performance = 'Low'

        data.append({
            'ad_id': f'AD_{i + 1:04d}',
            'ad_text': ad_text,
            'industry': industry,
            'platform': platform,
            'date': random_date.strftime('%Y-%m-%d'),
            'day_of_week': day_of_week,
            'hour': hour,
            'impressions': impressions,
            'clicks': clicks,
            'ctr': round(ctr, 4),
            'cpc': round(cpc, 2),
            'cost': round(cost, 2),
            'conversions': conversions,
            'cvr': round(cvr, 4),
            'revenue': round(revenue, 2),
            'roi': round(roi, 3),
            'sentiment_score': round(sentiment_score, 3),
            'performance': performance
        })

    return pd.DataFrame(data)


df = generate_ad_dataset(1000)

df.to_csv('../data/raw/ad_performance_data.csv', index=False)

print(f"✅ Dataset generated with {len(df)} ad campaigns!")
print(f"📊 Performance distribution:")
print(df['performance'].value_counts())
print(f"\n📈 Sample data:")
print(df.head())

print(f"\n📊 Quick Statistics:")
print(f"Average CTR: {df['ctr'].mean():.3f}")
print(f"Average CPC: ${df['cpc'].mean():.2f}")
print(f"Average ROI: {df['roi'].mean():.2f}")
print(f"Total Revenue: ${df['revenue'].sum():,.2f}")