FROM node:22-alpine AS dashboard-build
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
ARG VITE_API_BASE_URL=
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL}
RUN npm run build

FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY api/ ./api/
COPY src/ ./src/
COPY models/ ./models/
COPY reports/model_evaluation.json ./reports/model_evaluation.json
COPY data/processed/solarguard_hourly.csv.gz ./data/processed/solarguard_hourly.csv.gz
COPY --from=dashboard-build /web/dist ./frontend/dist
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
