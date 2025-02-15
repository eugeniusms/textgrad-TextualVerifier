# Tutorial

## Setup

Follow these steps to set up your development environment:

### 1. Create a Virtual Environment
```bash
python3 -m venv env
```

### 2. Activate the Virtual Environment
- **Mac/Linux**:
  ```bash
  source env/bin/activate
  ```
- **Windows**:
  ```bash
  env\Scripts\activate
  ```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install Package in Editable Mode
```bash
pip install -e .
```

### 5. Add .env File
```bash
GOOGLE_API_KEY=xxx
```

### 6. Test with Run The Example
```bash
python3 examples/code/tutorial_primitives.py
```

Your development environment is now ready to use!


## You Can Use Jupyter