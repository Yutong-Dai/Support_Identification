from scipy.io import savemat
import os 
ROOT = os.path.expanduser("~/db")
for db in ["a9a", "avazu-app.tr", "covtype", "kdda", "new20", "phishing", "rcv1", "real-sim", "url_combined", "w8a"]:
    Lip_path = f"{ROOT}/Lip/Lip-{db}.mat"
    if os.path.exists(Lip_path):
        print(f"{Lip_path} exists! Existing...")
    else:
        savemat(Lip_path, {"L": 0.25})
