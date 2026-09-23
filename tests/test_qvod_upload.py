import requests

BASE_URL = "http://127.0.0.1:5000"
headers = {"X-API-Key": "threatora-zero-trust"}

pcap_path = 'data/uploads/botnet-capture-20110816-qvod.pcap'
with open(pcap_path, 'rb') as f:
    r_up = requests.post(f"{BASE_URL}/api/upload", headers=headers, files={'file': ('botnet-capture-20110816-qvod.pcap', f, 'application/vnd.tcpdump.pcap')})

print('Upload status:', r_up.status_code)
data = r_up.json()
print('Status:', data.get('status'))
print('File label:', data.get('file_label'))
kpis = data.get('kpis', {})
print('KPIs Stage ID:', kpis.get('stage_id'), '| Stage Name:', kpis.get('stage_name'), '| Technique:', kpis.get('technique_id'), kpis.get('technique'))
print('Peak risk:', kpis.get('peak_risk_pct'), '%')
dist = data.get('attack_distribution', [])
total_dist_windows = sum(d.get('count', 0) for d in dist)
print('Total dynamic windows count:', total_dist_windows)
for d in dist:
    if d.get('count', 0) > 0:
        print(f"  [-] {d['name']}: {d['count']} windows ({d['percentage']}%) color={d['color']}")

pb = data.get('playbooks', [])
if pb:
    print('Playbook damage assessment:', pb[0].get('damage_assessment'))
