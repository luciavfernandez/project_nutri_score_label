import requests
import pandas as pd
import time
import os
from tqdm import tqdm
import json

def download_openfoodfacts_bulk(num_products=5000):
    """
    Download 5K diverse Open Food Facts products with retry logic + caching
    """
    
    # Cache file to resume if interrupted
    cache_file = 'openfoodfacts_cache.json'
    
    # Load existing cache
    if os.path.exists(cache_file):
        print("📂 Loading existing cache...")
        with open(cache_file, 'r') as f:
            all_products = json.load(f)
        print(f"   Loaded {len(all_products)} products from cache")
    else:
        all_products = []
    
    # Main search endpoint
    base_url = "https://world.openfoodfacts.org/cgi/search.pl"
    
    # Diverse category list for broad coverage
    categories = [
        "cookies", "bread", "beverages", "dairy", "meat", "cereals", 
        "pasta", "rice", "snacks", "chocolate", "ice-cream", "yogurt",
        "cheese", "butter", "jam", "honey", "nuts", "fruits", "vegetables",
        "ready-meals", "pizza", "burgers", "sausages", "fish"
    ]
    
    nutrition_grades = ['a', 'b', 'c', 'd', 'e']
    
    session = requests.Session()
    session.headers.update({'User-Agent': 'NutriScoreProject/1.0'})
    
    products_needed = num_products - len(all_products)
    print(f"🎯 Need {products_needed} more products (target: {num_products})")
    
    with tqdm(total=products_needed, desc="Fetching") as pbar:
        for category in categories:
            for grade in nutrition_grades:
                if len(all_products) >= num_products:
                    break
                    
                for page in range(1, 20):  # Max 20 pages per combo
                    if len(all_products) >= num_products:
                        break
                    
                    params = {
                        'search_terms': category,
                        'search_simple': 1,
                        'json': 1,
                        'page_size': 100,
                        'page': page,
                        'tagtype_0': 'nutrition_grades',
                        'tag_contains_0': 'contains',
                        'tag_0': grade,
                        'fields': ','.join([
                            'code', 'product_name', 'brands', 'categories_tags',
                            'nutrition_grades', 'ecoscore_grade', 'ecoscore_score',
                            'nutriments'  # Request the full nutriments object
                        ])
                    }
                    
                    try:
                        response = session.get(base_url, params=params, timeout=20)
                        response.raise_for_status()
                        
                        data = response.json()
                        products = data.get('products', [])
                        
                        if not products:
                            break  # No more products
                        
                        # Filter for complete nutrition data
                        valid_products = []
                        for product in products:
                            nutriments = product.get('nutriments', {})
                            
                            # REQUIRED: Must have KJ energy + all key nutrients
                            if (nutriments.get('energy-kj_100g') and 
                                nutriments.get('sugars_100g') and 
                                nutriments.get('salt_100g') and 
                                nutriments.get('saturated-fat_100g') and 
                                nutriments.get('proteins_100g') and 
                                nutriments.get('fiber_100g')):
                                
                                valid_products.append(product)
                        
                        all_products.extend(valid_products)
                        new_count = len(valid_products)
                        
                        pbar.update(new_count)
                        
                        # Save cache every 500 products
                        if len(all_products) % 500 == 0:
                            with open(cache_file, 'w') as f:
                                json.dump(all_products, f)
                            print(f"\n💾 Cached {len(all_products)} products")
                        
                        time.sleep(0.3)  # Rate limiting
                        
                    except Exception as e:
                        print(f"\n⚠️ Error {category}-{grade}-{page}: {str(e)[:50]}")
                        time.sleep(2)
                        continue
    
    # Save final dataset
    df = pd.DataFrame(all_products)
    
    # Clean and standardize
    df_clean = df[df['nutriments'].apply(lambda x: x.get('energy-kj_100g') is not None)].copy()
    
    # Extract nutriments to columns
    nutriments_df = pd.json_normalize(df_clean['nutriments'])
    df_final = pd.concat([df_clean[['code', 'product_name', 'nutrition_grades', 'ecoscore_grade']], nutriments_df], axis=1)
    
    # Rename to match your project
    column_mapping = {
        'energy-kj_100g': 'energy_kj',
        'sugars_100g': 'sugars_g', 
        'salt_100g': 'salt_g',
        'saturated-fat_100g': 'saturated_fat_g',
        'proteins_100g': 'proteins_g',
        'fiber_100g': 'fiber_g',
        'fruits_vegetables_nuts_100g': 'fruit_veg_pct'
    }
    
    df_final = df_final.rename(columns=column_mapping)
    
    # Select final columns
    final_columns = ['code', 'product_name', 'nutrition_grades', 'ecoscore_grade', 
                    'energy_kj', 'sugars_g', 'salt_g', 'saturated_fat_g', 
                    'proteins_g', 'fiber_g', 'fruit_veg_pct']
    
    df_final = df_final[final_columns].dropna(subset=['energy_kj'])
    
    # Remove exact duplicates
    df_final = df_final.drop_duplicates(subset=['product_name', 'energy_kj'])
    
    print(f"\n🎉 FINAL DATASET:")
    print(f"   Total products: {len(df_final):,}")
    print(f"   Nutri-Score distribution:")
    print(df_final['nutrition_grades'].value_counts())
    print(f"   Green-Score distribution:")
    print(df_final['ecoscore_grade'].value_counts())
    
    # Save all formats
    df_final.to_csv('nutriscore_5k_products.csv', index=False)
    df_final.to_excel('nutriscore_5k_products.xlsx', index=False)
    
    # Save raw cache too
    with open(cache_file, 'w') as f:
        json.dump(all_products, f)
    
    print(f"\n💾 Saved to:")
    print(f"   - nutriscore_5k_products.csv")
    print(f"   - nutriscore_5k_products.xlsx")
    print(f"   - openfoodfacts_cache.json (resume file)")
    
    return df_final

# RUN IT    
if __name__ == "__main__":
    df = download_openfoodfacts_bulk(5000)
    print("\n✅ Done! Ready for your Nutri-Score project!")
