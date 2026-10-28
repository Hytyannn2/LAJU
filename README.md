# LAJU

AI bus network designer for Kelantan. Full roadmap, design notes and benchmark results: [PLAN.txt](PLAN.txt).

```
# terminal 1
cd engine
.venv\Scripts\pip install -r requirements.txt   # first time only
.venv\Scripts\python kelantan.py                # first time only: downloads ~300 MB of map/population data
.venv\Scripts\python -m uvicorn server:app --port 8000 --reload

# terminal 2
cd web
npm run dev        # http://localhost:3000
```

Data: © OpenStreetMap contributors (ODbL), Kontur Population (CC BY 4.0), OpenDOSM (CC BY 4.0), Microsoft Global ML Building Footprints (ODbL), hospital beds (Ministry of Health via data.gov.my, CC BY 4.0), school enrolment (Ministry of Education via OpenDOSM, CC BY 4.0). Satellite map: Esri World Imagery.
