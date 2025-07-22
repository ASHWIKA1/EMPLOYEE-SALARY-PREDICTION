import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

def create_synthetic_salary_data():
    """Create synthetic salary data for Indian market with realistic salary ranges"""
    np.random.seed(42)
    n_samples = 10000
    
    # Education levels with corresponding salary multipliers
    education_levels = ['High School', 'Some College', 'Bachelors', 'Masters', 'PhD']
    education_salary_multipliers = [1.0, 1.3, 1.8, 2.5, 3.2]
    
    # Job titles with base salaries (in INR)
    job_titles = ['Clerk', 'Sales Representative', 'Software Developer', 'Manager', 'Engineer', 
                  'Teacher', 'Nurse', 'Accountant', 'Designer', 'Analyst', 'Consultant', 'Director']
    base_salaries = [250000, 350000, 600000, 800000, 550000, 400000, 450000, 500000, 450000, 550000, 700000, 1200000]
    
    # Work locations with cost of living multipliers
    locations = ['Rural', 'Suburban', 'Urban', 'Metropolitan']
    location_multipliers = [0.8, 1.0, 1.3, 1.6]
    
    # Generate synthetic data
    data = []
    
    for _ in range(n_samples):
        age = np.random.randint(18, 66)
        experience = np.random.randint(0, 41)
        education_idx = np.random.randint(0, len(education_levels))
        job_idx = np.random.randint(0, len(job_titles))
        location_idx = np.random.randint(0, len(locations))
        gender = np.random.choice(['Male', 'Female'])
        
        # Calculate base salary
        base_salary = base_salaries[job_idx]
        
        # Apply multipliers
        education_multiplier = education_salary_multipliers[education_idx]
        location_multiplier = location_multipliers[location_idx]
        experience_multiplier = 1 + (experience * 0.05)  # 5% increase per year
        age_multiplier = 1 + (age - 25) * 0.01 if age > 25 else 0.8  # Age factor
        
        # Calculate final salary with some randomness
        salary = base_salary * education_multiplier * location_multiplier * experience_multiplier * age_multiplier
        salary *= np.random.uniform(0.8, 1.2)  # Add some randomness
        
        # Ensure salary is within realistic bounds
        salary = max(150000, min(salary, 5000000))
        
        data.append({
            'age': age,
            'education_level': education_levels[education_idx],
            'years_experience': experience,
            'job_title': job_titles[job_idx],
            'work_location': locations[location_idx],
            'gender': gender,
            'salary': salary
        })
    
    return pd.DataFrame(data)

def train_salary_model():
    """Train the salary prediction model"""
    print("Creating synthetic salary data...")
    data = create_synthetic_salary_data()
    
    print("Preprocessing data...")
    # Create encoders
    le_education = LabelEncoder()
    le_job = LabelEncoder()
    le_location = LabelEncoder()
    le_gender = LabelEncoder()
    
    # Encode categorical variables
    data['education_encoded'] = le_education.fit_transform(data['education_level'])
    data['job_encoded'] = le_job.fit_transform(data['job_title'])
    data['location_encoded'] = le_location.fit_transform(data['work_location'])
    data['gender_encoded'] = le_gender.fit_transform(data['gender'])
    
    # Prepare features
    X = data[['age', 'education_encoded', 'years_experience', 'job_encoded', 
              'location_encoded', 'gender_encoded']]
    y = data['salary']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Training Random Forest model...")
    # Train Random Forest model
    model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    accuracy = 100 - (mae / y_test.mean() * 100)
    
    print(f"Model Performance:")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Absolute Error: ₹{mae:,.0f}")
    print(f"Accuracy: {accuracy:.2f}%")
    
    # Save model and encoders
    joblib.dump(model, 'salary_model.pkl')
    joblib.dump(scaler, 'salary_scaler.pkl')
    joblib.dump(le_education, 'education_encoder.pkl')
    joblib.dump(le_job, 'job_encoder.pkl')
    joblib.dump(le_location, 'location_encoder.pkl')
    joblib.dump(le_gender, 'gender_encoder.pkl')
    
    print("Model and encoders saved successfully!")
    
    return model, scaler, le_education, le_job, le_location, le_gender

if __name__ == "__main__":
    train_salary_model() 