#!/usr/bin/env python
"""
Run script for the Idiom App application
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("idiomapp.main:app", host="0.0.0.0", port=8001, reload=True) 