from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel
from predict import predict_data
import uvicorn

app = FastAPI()

# Define schema for breast cancer dataset (30 features)
class BreastCancerData(BaseModel):
    mean_radius: float
    mean_texture: float
    mean_perimeter: float
    mean_area: float
    mean_smoothness: float
    mean_compactness: float
    mean_concavity: float
    mean_concave_points: float
    mean_symmetry: float
    mean_fractal_dimension: float
    radius_error: float
    texture_error: float
    perimeter_error: float
    area_error: float
    smoothness_error: float
    compactness_error: float
    concavity_error: float
    concave_points_error: float
    symmetry_error: float
    fractal_dimension_error: float
    worst_radius: float
    worst_texture: float
    worst_perimeter: float
    worst_area: float
    worst_smoothness: float
    worst_compactness: float
    worst_concavity: float
    worst_concave_points: float
    worst_symmetry: float
    worst_fractal_dimension: float

class BreastCancerResponse(BaseModel):
    response: int

@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    return {"status": "healthy"}

@app.post("/predict", response_model=BreastCancerResponse)
async def predict_cancer(features: BreastCancerData):
    try:
        # Convert input features into the order expected by sklearn’s model
        feature_vector = [[
            features.mean_radius,
            features.mean_texture,
            features.mean_perimeter,
            features.mean_area,
            features.mean_smoothness,
            features.mean_compactness,
            features.mean_concavity,
            features.mean_concave_points,
            features.mean_symmetry,
            features.mean_fractal_dimension,
            features.radius_error,
            features.texture_error,
            features.perimeter_error,
            features.area_error,
            features.smoothness_error,
            features.compactness_error,
            features.concavity_error,
            features.concave_points_error,
            features.symmetry_error,
            features.fractal_dimension_error,
            features.worst_radius,
            features.worst_texture,
            features.worst_perimeter,
            features.worst_area,
            features.worst_smoothness,
            features.worst_compactness,
            features.worst_concavity,
            features.worst_concave_points,
            features.worst_symmetry,
            features.worst_fractal_dimension
        ]]
        prediction = predict_data(feature_vector)
        return BreastCancerResponse(response=int(prediction[0]))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=4000,
        reload=True
    )