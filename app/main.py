from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.routes import router # Import your routes

app = FastAPI()

# Mount the static directory for serving CSS and JS files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Include the router in the FastAPI app
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)