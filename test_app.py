from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"message": "✅ FastAPI is working!"}

@app.get("/hello")
def hello():
    return {"message": "Hello from your API!"}
