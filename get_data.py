import requests
import pandas as pd
import time
import os
from tqdm import tqdm
import json

def calculate_nutriscore_points(energy_kj, sugars, sat_fat, salt, proteins, fiber, fvl_percent):
    """
    Calculate Nutri-Score points based on the algorithm from the PDF.
    Returns the nutritional score (N - P)
    """
    negative_points = 0
    positive_points = 0
    
    # Negative points for Energy (kJ/100g)
    if energy_kj > 3350:
        negative_points += 10
    elif energy_kj > 3015:
        negative_points += 9
    elif energy_kj > 2680:
        negative_points += 8
    elif energy_kj > 2345:
        negative_points += 7
    elif energy_kj > 2010:
        negative_points += 6
    elif energy_kj > 1675:
        negative_points += 5
    elif energy_kj > 1340:
        negative_points += 4
    elif energy_kj > 1005:
        negative_points += 3
    elif energy_kj > 670:
        negative_points += 2
    elif energy_kj > 335:
        negative_points += 1
    
    # Negative points for Sugars (g/100g)
    if sugars > 51:
        negative_points += 15
    elif sugars > 48:
        negative_points += 14
    elif sugars > 44:
        negative_points += 13
    elif sugars > 41:
        negative_points += 12
    elif sugars > 37:
        negative_points += 11
    elif sugars > 34:
        negative_points += 10
    elif sugars > 31:
        negative_points += 9
    elif sugars > 27:
        negative_points += 8
    elif sugars > 24:
        negative_points += 7
    elif sugars > 20:
        negative_points += 6
    elif sugars > 17:
        negative_points += 5
    elif sugars > 14:
        negative_points += 4
    elif sugars > 10:
        negative_points += 3
    elif sugars > 6.8:
        negative_points += 2
    elif sugars > 3.4:
        negative_points += 1
    
    # Negative points for Saturated fat (g/100g)
    if sat_fat > 10:
        negative_points += 10
    elif sat_fat > 9:
        negative_points += 9
    elif sat_fat > 8:
        negative_points += 8
    elif sat_fat > 7:
        negative_points += 7
    elif sat_fat > 6:
        negative_points += 6
    elif sat_fat > 5:
        negative_points += 5
    elif sat_fat > 4:
        negative_points += 4
    elif sat_fat > 3:
        negative_points += 3
    elif sat_fat > 2:
        negative_points += 2
    elif sat_fat > 1:
        negative_points += 1
    
    # Negative points for Salt (g/100g)
    if salt > 4:
        negative_points += 20
    elif salt > 3.8:
        negative_points += 19
    elif salt > 3.6:
        negative_points += 18
    elif salt > 3.4:
        negative_points += 17
    elif salt > 3.2:
        negative_points += 16
    elif salt > 3:
        negative_points += 15
    elif salt > 2.8:
        negative_points += 14
    elif salt > 2.6:
        negative_points += 13
    elif salt > 2.4:
        negative_points += 12
    elif salt > 2.2:
        negative_points += 11
    elif salt > 2:
        negative_points += 10
    elif salt > 1.8:
        negative_points += 9
    elif salt > 1.6:
        negative_points += 8
    elif salt > 1.4:
        negative_points += 7
    elif salt > 1.2:
        negative_points += 6
    elif salt > 1:
        negative_points += 5
    elif salt > 0.8:
        negative_points += 4
    elif salt > 0.6:
        negative_points += 3
    elif salt > 0.4:
        negative_points += 2
    elif salt > 0.2:
        negative_points += 1
    
    # Positive points for Proteins (g/100g)
    protein_points = 0
    if proteins > 17:
        protein_points = 7
    elif proteins > 14:
        protein_points = 6
    elif proteins > 12:
        protein_points = 5
    elif proteins > 9.6:
        protein_points = 4
    elif proteins > 7.2:
        protein_points = 3
    elif proteins > 4.8:
        protein_points = 2
    elif proteins > 2.4:
        protein_points = 1
    
    # Positive points for Fiber (g/100g)
    if fiber > 7.4:
        positive_points += 7
    elif fiber > 6.3:
        positive_points += 6
    elif fiber > 5.2:
        positive_points += 5
    elif fiber > 4.1:
        positive_points += 4
    elif fiber > 3.0:
        positive_points += 3
    elif fiber > 0:
        positive_points += int(fiber / 0.9)  # Approximate
    
    # Positive points for Fruits/Vegetables/Legumes (%)
    if fvl_percent > 80:
        positive_points += 5
    elif fvl_percent > 60:
        positive_points += 2
    elif fvl_percent > 40:
        positive_points += 1
    
    # Protein points calculation rule:
    # If negative >= 11 and fvl_percent <= 80, proteins don't count
    if negative_points >= 11 and fvl_percent <= 80:
        nutri_score = negative_points - positive_points
    else:
        nutri_score = negative_points - positive_points - protein_points
    
    return nutri_score

def download_openfoodfacts_bulk(num_products=5000):
    """
    Download Open Food Facts products with complete nutritional and score data
    Output matches the structure of food_database.csv
    """
    
    cache_file = 'openfoodfacts_cache.json'
    
    if os.path.exists(cache_file):
        print("Loading existing cache...")
        with open(cache_file, 'r') as f:
            all_products = json.load(f)
        print(f"Loaded {len(all_products)} products from cache")
    else:
        all_products = []
    
    base_url = "https://world.openfoodfacts.org/cgi/search.pl"
    
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
    print(f"Need {products_needed} more products (target: {num_products})")

    last_save_count = len(all_products)

    try:
        with tqdm(total=products_needed, desc="Fetching") as pbar:
            for category in categories:
                for grade in nutrition_grades:
                    if len(all_products) >= num_products:
                        break

                    for page in range(1, 20):
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
                                'code', 'product_name', 'brands', 'categories', 'categories_tags',
                                'nutrition_grades', 'nutrition_score_debug', 
                                'ecoscore_grade', 'ecoscore_score',
                                'nutriments', 'url'
                            ])
                        }

                        try:
                            response = session.get(base_url, params=params, timeout=20)
                            response.raise_for_status()

                            data = response.json()
                            products = data.get('products', [])

                            if not products:
                                break

                            valid_products = []
                            for product in products:
                                nutriments = product.get('nutriments', {})

                                if (nutriments.get('energy-kj_100g') and
                                    nutriments.get('sugars_100g') is not None and
                                    nutriments.get('salt_100g') is not None and
                                    nutriments.get('saturated-fat_100g') is not None and
                                    nutriments.get('proteins_100g') is not None and
                                    nutriments.get('fiber_100g') is not None):

                                    valid_products.append(product)

                            all_products.extend(valid_products)
                            new_count = len(valid_products)

                            pbar.update(new_count)

                            if len(all_products) - last_save_count >= 200:
                                with open(cache_file, 'w') as f:
                                    json.dump(all_products, f)
                                print(f"\nCached {len(all_products)} products")
                                last_save_count = len(all_products)

                            time.sleep(0.3)

                        except Exception as e:
                            print(f"\nError {category}-{grade}-{page}: {str(e)[:50]}")
                            time.sleep(2)
                            continue

    except KeyboardInterrupt:
        print(f"\n\nInterrupted! Saving progress...")
    finally:
        if len(all_products) > 0:
            with open(cache_file, 'w') as f:
                json.dump(all_products, f)
            print(f"Saved {len(all_products)} products to cache")
    
    print(f"\nProcessing {len(all_products)} products into final dataset...")
    
    processed_rows = []
    skipped_count = {
        'missing_nutrients': 0,
        'empty_product_name': 0,
        'unknown_green_score': 0,
        'empty_nutri_label': 0,
        'invalid_values': 0
    }
    
    for product in all_products:
        try:
            nutriments = product.get('nutriments', {})
            
            # Extract all required nutrient values
            energy_kj = nutriments.get('energy-kj_100g') or nutriments.get('energy_100g')
            sugars = nutriments.get('sugars_100g')
            sat_fat = nutriments.get('saturated-fat_100g')
            salt = nutriments.get('salt_100g')
            proteins = nutriments.get('proteins_100g')
            fiber = nutriments.get('fiber_100g')
            fvl_percent = nutriments.get('fruits-vegetables-nuts-estimate-from-ingredients_100g', 0)
            
            # Skip if any required nutrient is missing or None
            if energy_kj is None or sugars is None or sat_fat is None or salt is None or proteins is None or fiber is None:
                skipped_count['missing_nutrients'] += 1
                continue
            
            # Validate that all values are numeric and non-negative
            try:
                energy_kj = float(energy_kj)
                sugars = float(sugars)
                sat_fat = float(sat_fat)
                salt = float(salt)
                proteins = float(proteins)
                fiber = float(fiber)
                fvl_percent = float(fvl_percent) if fvl_percent else 0
                
                # Check for negative values (invalid data)
                if any(val < 0 for val in [energy_kj, sugars, sat_fat, salt, proteins, fiber]):
                    skipped_count['invalid_values'] += 1
                    continue
                    
            except (ValueError, TypeError):
                skipped_count['invalid_values'] += 1
                continue
            
            # Get and validate product name
            product_name = str(product.get('product_name', '')).strip()
            if not product_name or product_name == 'nan' or product_name == 'None':
                skipped_count['empty_product_name'] += 1
                continue
            
            # Get and validate Nutri-Score label
            nutri_grade = product.get('nutrition_grades', '')
            if isinstance(nutri_grade, str):
                nutri_grade = nutri_grade.strip().upper()
            else:
                nutri_grade = ''
            
            # Skip if Nutri-Score label is empty or invalid
            if not nutri_grade or nutri_grade not in ['A', 'B', 'C', 'D', 'E']:
                skipped_count['empty_nutri_label'] += 1
                continue
            
            # Get and validate Eco-Score (Green Score) label
            eco_grade = product.get('ecoscore_grade', '')
            if isinstance(eco_grade, str):
                eco_grade = eco_grade.strip().upper()
            else:
                eco_grade = ''
            
            # Skip if Green Score is empty, UNKNOWN, or invalid
            if not eco_grade or eco_grade == 'UNKNOWN' or eco_grade == 'NOT-APPLICABLE' or eco_grade not in ['A', 'B', 'C', 'D', 'E']:
                skipped_count['unknown_green_score'] += 1
                continue
            
            # Get and validate Eco-Score value
            eco_score_value = product.get('ecoscore_score', '')
            if eco_score_value == '' or eco_score_value is None:
                skipped_count['unknown_green_score'] += 1
                continue
            
            try:
                eco_score_value = float(eco_score_value)
                if eco_score_value < 0 or eco_score_value > 100:
                    skipped_count['unknown_green_score'] += 1
                    continue
            except (ValueError, TypeError):
                skipped_count['unknown_green_score'] += 1
                continue
            
            # Get brands and categories (optional but clean them)
            brands = str(product.get('brands', '')).strip()
            if brands in ['nan', 'None', '']:
                brands = ''
            
            categories = str(product.get('categories', '')).strip()
            if categories in ['nan', 'None', '']:
                categories = ''
            
            # Calculate Nutri-Score value using our algorithm
            nutri_score_value = calculate_nutriscore_points(
                energy_kj, sugars, sat_fat, salt, proteins, fiber, fvl_percent
            )
            
            # Generate URL if missing
            product_url = product.get('url', '')
            if not product_url and product.get('code'):
                product_url = f"https://world.openfoodfacts.org/product/{product.get('code')}"
            
            # Create clean row
            row = {
                'product_name': product_name,
                'brands': brands,
                'categories': categories,
                'energy_100g': energy_kj,
                'saturated_fat_100g': sat_fat,
                'sugars_100g': sugars,
                'salt_100g': salt,
                'proteins_100g': proteins,
                'fiber_100g': fiber,
                'fvl_percent': fvl_percent,
                'nutri_score_value': nutri_score_value,
                'nutri_score_label': nutri_grade,
                'green_score_value': eco_score_value,
                'green_score_label': eco_grade,
                'url': product_url
            }
            
            processed_rows.append(row)
                
        except Exception as e:
            skipped_count['invalid_values'] += 1
            continue
    
    # Print summary of skipped products
    print(f"\nData cleaning summary:")
    print(f"  Products with missing nutrients: {skipped_count['missing_nutrients']}")
    print(f"  Products with empty names: {skipped_count['empty_product_name']}")
    print(f"  Products with unknown/missing green score: {skipped_count['unknown_green_score']}")
    print(f"  Products with empty/invalid nutri label: {skipped_count['empty_nutri_label']}")
    print(f"  Products with invalid values: {skipped_count['invalid_values']}")
    print(f"  Total skipped: {sum(skipped_count.values())}")
    print(f"  Valid products: {len(processed_rows)}")
    
    df_final = pd.DataFrame(processed_rows)
    
    print(f"\nCreated DataFrame with {len(df_final)} rows")
    
    if df_final.empty:
        print("Warning: No valid products found in the dataset")
        return df_final
    
    # Additional data cleaning on the DataFrame
    initial_count = len(df_final)
    
    # Remove rows with empty product names
    if 'product_name' in df_final.columns:
        df_final = df_final[df_final['product_name'].str.strip() != '']
        print(f"After filtering empty names: {len(df_final)} rows")
    
    # Remove rows with invalid Nutri-Score labels
    if 'nutri_score_label' in df_final.columns:
        df_final = df_final[df_final['nutri_score_label'].isin(['A', 'B', 'C', 'D', 'E'])]
        print(f"After filtering invalid Nutri-Score labels: {len(df_final)} rows")
    
    # Remove rows with invalid Green-Score labels
    if 'green_score_label' in df_final.columns:
        df_final = df_final[df_final['green_score_label'].isin(['A', 'B', 'C', 'D', 'E'])]
        print(f"After filtering invalid Green-Score labels: {len(df_final)} rows")
    
    # Remove rows with null or negative nutritional values
    numeric_columns = ['energy_100g', 'saturated_fat_100g', 'sugars_100g', 
                      'salt_100g', 'proteins_100g', 'fiber_100g']
    for col in numeric_columns:
        if col in df_final.columns:
            df_final = df_final[df_final[col].notna()]
            df_final = df_final[df_final[col] >= 0]
    print(f"After filtering null/negative values: {len(df_final)} rows")
    
    # Remove duplicates based on product_name and energy_100g
    if 'product_name' in df_final.columns and 'energy_100g' in df_final.columns:
        before_dedup = len(df_final)
        df_final = df_final.drop_duplicates(subset=['product_name', 'energy_100g'])
        print(f"After removing duplicates: {len(df_final)} rows (removed {before_dedup - len(df_final)} duplicates)")
    
    # Remove products with extreme outlier values (likely data errors)
    if 'energy_100g' in df_final.columns:
        # Energy should be reasonable (not > 4000 kJ per 100g, which is extremely high)
        df_final = df_final[df_final['energy_100g'] <= 4000]
    
    if 'sugars_100g' in df_final.columns:
        # Sugars cannot exceed 100g per 100g
        df_final = df_final[df_final['sugars_100g'] <= 100]
    
    if 'salt_100g' in df_final.columns:
        # Salt should be reasonable (not > 10g per 100g)
        df_final = df_final[df_final['salt_100g'] <= 10]
    
    if 'saturated_fat_100g' in df_final.columns:
        # Saturated fat cannot exceed 100g per 100g
        df_final = df_final[df_final['saturated_fat_100g'] <= 100]
    
    if 'proteins_100g' in df_final.columns:
        # Proteins cannot exceed 100g per 100g
        df_final = df_final[df_final['proteins_100g'] <= 100]
    
    if 'fiber_100g' in df_final.columns:
        # Fiber cannot exceed 100g per 100g
        df_final = df_final[df_final['fiber_100g'] <= 100]
    
    if 'fvl_percent' in df_final.columns:
        # FVL percent should be 0-100
        df_final = df_final[df_final['fvl_percent'] <= 100]
    
    print(f"After removing outliers: {len(df_final)} rows")
    print(f"Total rows removed during cleaning: {initial_count - len(df_final)}")
    
    column_order = [
        'product_name', 'brands', 'categories', 'energy_100g', 
        'saturated_fat_100g', 'sugars_100g', 'salt_100g', 
        'proteins_100g', 'fiber_100g', 'fvl_percent',
        'nutri_score_value', 'nutri_score_label', 
        'green_score_value', 'green_score_label', 'url'
    ]
    
    existing_columns = [col for col in column_order if col in df_final.columns]
    df_final = df_final[existing_columns]
    
    print(f"\nFINAL DATASET:")
    print(f"Total products: {len(df_final):,}")
    
    if 'nutri_score_label' in df_final.columns:
        print(f"\nNutri-Score distribution:")
        print(df_final['nutri_score_label'].value_counts().sort_index())
    
    if 'green_score_label' in df_final.columns:
        print(f"\nEco-Score distribution:")
        print(df_final['green_score_label'].value_counts().sort_index())
    
    df_final.to_csv('food_database.csv', index=False, encoding='utf-8-sig')
    df_final.to_excel('food_database.xlsx', index=False)
    
    with open(cache_file, 'w') as f:
        json.dump(all_products, f)
    
    print(f"\nSaved to:")
    print(f"- food_database.csv")
    print(f"- food_database.xlsx")
    print(f"- openfoodfacts_cache.json (resume file)")
    
    return df_final

if __name__ == "__main__":
    df = download_openfoodfacts_bulk(5000)
    print("\nDone! Ready for your Nutri-Score project!")