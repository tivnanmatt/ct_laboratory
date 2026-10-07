"""AssetStore / Session / run_job round trip (no GPU)."""
import tempfile
from ct_laboratory.workflow import AssetStore, Session, run_job
d = tempfile.mkdtemp(); st = AssetStore(d + '/assets'); f = d + '/geo.pt'; open(f, 'wb').write(b'hello')
a = st.put_upload('geometry', [f]); assert st.put_upload('geometry', [f]).id == a.id
stage = st.stage_dir('eigen'); open(stage + '/basis.pt', 'wb').write(b'x')
c = st.put_computed('eigen', stage, 'eigen.compute', '1', {'geometry': a.id}, {'k': 32})
assert st.lookup_recipe('eigen', 'eigen.compute', '1', {'geometry': a.id}, {'k': 32}).id == c.id
assert st.lookup_recipe('eigen', 'eigen.compute', '1', {'geometry': a.id}, {'k': 64}) is None
s = Session(d + '/scan/recon/sessions/t', st); s.set('geometry', a.id)
j = run_job(s, 'eigen.compute', {'k': 32}, lambda cfg, sess, job: {'outputs': {'eigen@64': c.id}})
assert j.status['state'] == 'COMPLETED' and s.get('eigen@64') == c.id and s.resolve('@eigen@64') == c.id
print('PASS workflow store/session/job')
