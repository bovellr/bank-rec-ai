# Debug the 'bool' object has no attribute 'item' error

import pandas as pd
import numpy as np
import sys
sys.path.insert(0, 'app')

def debug_bool_error():
    """Debug the specific boolean error in feature generation."""
    
    print("🔍 Debugging boolean type error...")
    
    # Create the exact same test data
    bank_df = pd.DataFrame({
        'Amount': [100.00, -50.00, 200.00, 75.50],
        'Date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04'],
        'Description': ['Payment to ABC Corp', 'ATM Withdrawal', 'Salary Credit', 'Grocery Store']
    })
    
    erp_df = pd.DataFrame({
        'Amount': [100.00, 200.00, 80.00, -45.00],
        'Date': ['2024-01-01', '2024-01-03', '2024-01-05', '2024-01-02'],
        'Description': ['ABC Corp Payment', 'Salary Payment', 'Store Purchase', 'Cash Withdrawal']
    })
    
    print("📊 Data types:")
    print("Bank DataFrame dtypes:")
    print(bank_df.dtypes)
    print("\nERP DataFrame dtypes:")
    print(erp_df.dtypes)
    
    # Test each step of feature generation manually
    print("\n🧪 Testing feature generation step by step...")
    
    # Get first pair (0, 0)
    i, j = 0, 0
    b = bank_df.iloc[i]
    e = erp_df.iloc[j]
    
    print(f"\nTesting pair ({i}, {j}):")
    print(f"Bank row: {b.to_dict()}")
    print(f"ERP row: {e.to_dict()}")
    
    try:
        # Test date difference calculation
        print("\n📅 Testing date operations...")
        bank_date = pd.to_datetime(b['Date'])
        erp_date = pd.to_datetime(e['Date'])
        print(f"Bank date: {bank_date} (type: {type(bank_date)})")
        print(f"ERP date: {erp_date} (type: {type(erp_date)})")
        
        date_diff_raw = bank_date - erp_date
        print(f"Date diff raw: {date_diff_raw} (type: {type(date_diff_raw)})")
        
        # This is likely where the error occurs
        try:
            date_diff = abs((pd.Timedelta(date_diff_raw)).days)
            print(f"✅ Date diff: {date_diff}")
        except Exception as e:
            print(f"❌ Date diff error: {e}")
            # Alternative calculation
            date_diff = abs((bank_date - erp_date).days)
            print(f"✅ Alternative date diff: {date_diff}")
        
        # Test amount operations
        print("\n💰 Testing amount operations...")
        amount_diff = abs(b['Amount'] - e['Amount'])
        print(f"✅ Amount diff: {amount_diff}")
        
        # Test signed amount match - this is another likely culprit
        print("\n🔢 Testing signed amount match...")
        b_positive = b['Amount'] > 0
        e_positive = e['Amount'] > 0
        print(f"Bank positive: {b_positive} (type: {type(b_positive)})")
        print(f"ERP positive: {e_positive} (type: {type(e_positive)})")
        
        # This comparison might cause the bool error
        try:
            signed_match_bool = (b_positive == e_positive)
            print(f"Signed match bool: {signed_match_bool} (type: {type(signed_match_bool)})")
            
            signed_match = int(signed_match_bool)
            print(f"✅ Signed match: {signed_match}")
        except Exception as exc:
            print(f"❌ Signed match error: {exc}")
            # Alternative approach
            signed_match = 1 if (b['Amount'] > 0) == (e['Amount'] > 0) else 0
            print(f"✅ Alternative signed match: {signed_match}")
        
        # Test same day calculation
        print("\n📆 Testing same day...")
        try:
            same_day_bool = (bank_date == erp_date)
            print(f"Same day bool: {same_day_bool} (type: {type(same_day_bool)})")
            
            same_day = int(same_day_bool)
            print(f"✅ Same day: {same_day}")
        except Exception as e:
            print(f"❌ Same day error: {e}")
            # Alternative
            same_day = 1 if bank_date.date() == erp_date.date() else 0
            print(f"✅ Alternative same day: {same_day}")
        
        # Test description similarity
        print("\n📝 Testing description similarity...")
        from rapidfuzz.fuzz import token_sort_ratio
        desc_sim = token_sort_ratio(str(b['Description']), str(e['Description']))
        print(f"✅ Description similarity: {desc_sim}")
        
        print("\n✅ All individual operations successful!")
        
    except Exception as e:
        print(f"❌ Error in step-by-step testing: {e}")
        import traceback
        traceback.print_exc()

def test_boolean_conversions():
    """Test different ways to convert boolean to int."""
    
    print("\n🧪 Testing boolean conversion methods...")
    
    # Create test boolean values
    test_values = [
        True,
        False,
        np.bool_(True),
        np.bool_(False),
        pd.Series([True])[0],
        pd.Series([False])[0]
    ]
    
    for i, val in enumerate(test_values):
        print(f"\nTest {i+1}: {val} (type: {type(val)})")
        
        # Method 1: int()
        try:
            result1 = int(val)
            print(f"  int(val): {result1} ✅")
        except Exception as e:
            print(f"  int(val): {e} ❌")
        
        # Method 2: bool() then int()
        try:
            result2 = int(bool(val))
            print(f"  int(bool(val)): {result2} ✅")
        except Exception as e:
            print(f"  int(bool(val)): {e} ❌")
        
        # Method 3: Conditional
        try:
            result3 = 1 if val else 0
            print(f"  1 if val else 0: {result3} ✅")
        except Exception as e:
            print(f"  1 if val else 0: {e} ❌")

def create_fixed_generate_features():
    """Create a fixed version of generate_features."""
    
    print("\n🔧 Creating fixed generate_features function...")
    
    fixed_code = '''
def generate_features_fixed(bank_df: pd.DataFrame, erp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fixed version of generate_features with proper type handling.
    """
    import pandas as pd
    from rapidfuzz.fuzz import token_sort_ratio
    
    # Quick validation
    if bank_df.empty or erp_df.empty:
        return pd.DataFrame()

    bank_df = bank_df.copy()
    erp_df = erp_df.copy()

    # Ensure date columns are datetime
    bank_df['Date'] = pd.to_datetime(bank_df['Date'], errors='coerce')
    erp_df['Date'] = pd.to_datetime(erp_df['Date'], errors='coerce')

    features = []
    
    for i, b in bank_df.iterrows():
        for j, e in erp_df.iterrows():
            try:
                # Safe amount handling
                b_amount = float(b['Amount']) if pd.notna(b['Amount']) else 0.0
                e_amount = float(e['Amount']) if pd.notna(e['Amount']) else 0.0
                
                # Safe date handling
                b_date = b['Date']
                e_date = e['Date']
                
                if pd.notna(b_date) and pd.notna(e_date):
                    # Fixed date difference calculation
                    date_delta = b_date - e_date
                    date_diff = abs(date_delta.days)
                    
                    # Fixed same day calculation
                    same_day = 1 if b_date.date() == e_date.date() else 0
                else:
                    date_diff = 9999
                    same_day = 0
                
                # Fixed signed amount match
                b_positive = b_amount > 0
                e_positive = e_amount > 0
                signed_match = 1 if b_positive == e_positive else 0
                
                # Amount difference
                amount_diff = abs(b_amount - e_amount)
                
                # Description similarity
                b_desc = str(b['Description']) if pd.notna(b['Description']) else ""
                e_desc = str(e['Description']) if pd.notna(e['Description']) else ""
                desc_sim = float(token_sort_ratio(b_desc, e_desc))
                
                features.append({
                    'bank_index': i,
                    'erp_index': j,
                    'amount_difference': amount_diff,
                    'date_difference': date_diff,
                    'description_similarity': desc_sim,
                    'signed_amount_match': signed_match,
                    'same_day': same_day
                })
                
            except Exception as ex:
                print(f"Feature generation failed for pair ({i}, {j}): {ex}")
                continue
   
    return pd.DataFrame(features)
'''
    
    print("💾 Fixed function created. Key changes:")
    print("1. ✅ Explicit float() conversions for amounts")
    print("2. ✅ Safe date.date() comparison for same_day")
    print("3. ✅ Simplified boolean to int conversion")
    print("4. ✅ Direct date difference without pd.Timedelta wrapper")
    print("5. ✅ Explicit float() conversion for description similarity")
    
    return fixed_code

def test_fixed_function():
    """Test the fixed function."""
    
    print("\n🧪 Testing fixed function...")
    
    # Define the fixed function locally
    def generate_features_fixed(bank_df, erp_df):
        import pandas as pd
        from rapidfuzz.fuzz import token_sort_ratio
        
        if bank_df.empty or erp_df.empty:
            return pd.DataFrame()

        bank_df = bank_df.copy()
        erp_df = erp_df.copy()

        bank_df['Date'] = pd.to_datetime(bank_df['Date'], errors='coerce')
        erp_df['Date'] = pd.to_datetime(erp_df['Date'], errors='coerce')

        features = []
        
        for i, b in bank_df.iterrows():
            for j, e in erp_df.iterrows():
                try:
                    b_amount = float(b['Amount']) if pd.notna(b['Amount']) else 0.0
                    e_amount = float(e['Amount']) if pd.notna(e['Amount']) else 0.0
                    
                    b_date = b['Date']
                    e_date = e['Date']
                    
                    if pd.notna(b_date) and pd.notna(e_date):
                        date_delta = b_date - e_date
                        date_diff = abs(date_delta.days)
                        same_day = 1 if b_date.date() == e_date.date() else 0
                    else:
                        date_diff = 9999
                        same_day = 0
                    
                    b_positive = b_amount > 0
                    e_positive = e_amount > 0
                    signed_match = 1 if b_positive == e_positive else 0
                    
                    amount_diff = abs(b_amount - e_amount)
                    
                    b_desc = str(b['Description']) if pd.notna(b['Description']) else ""
                    e_desc = str(e['Description']) if pd.notna(e['Description']) else ""
                    desc_sim = float(token_sort_ratio(b_desc, e_desc))
                    
                    features.append({
                        'bank_index': i,
                        'erp_index': j,
                        'amount_difference': amount_diff,
                        'date_difference': date_diff,
                        'description_similarity': desc_sim,
                        'signed_amount_match': signed_match,
                        'same_day': same_day
                    })
                    
                except Exception as ex:
                    print(f"Feature generation failed for pair ({i}, {j}): {ex}")
                    continue
       
        return pd.DataFrame(features)
    
    # Test with sample data
    bank_df = pd.DataFrame({
        'Amount': [100.00, -50.00],
        'Date': ['2024-01-01', '2024-01-02'],
        'Description': ['Payment to ABC Corp', 'ATM Withdrawal']
    })
    
    erp_df = pd.DataFrame({
        'Amount': [100.00, -45.00],
        'Date': ['2024-01-01', '2024-01-02'],
        'Description': ['ABC Corp Payment', 'Cash Withdrawal']
    })
    
    try:
        features = generate_features_fixed(bank_df, erp_df)
        print(f"✅ Fixed function works! Generated {len(features)} features")
        print("Features:")
        print(features)
        return True
    except Exception as e:
        print(f"❌ Fixed function still has issues: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_bool_error()
    test_boolean_conversions()
    create_fixed_generate_features()
    test_fixed_function()