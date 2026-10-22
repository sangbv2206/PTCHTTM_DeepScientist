import uvicorn


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  DeepScientist Research Workspace")
    print("  Mo trinh duyet tai: http://localhost:8000 hoac http://127.0.0.1:8000")
    print("=" * 60 + "\n")
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=False)
