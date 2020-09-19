import pandas as pd 

DIR = "News_dataset/"
FNS = ["True.csv", "Fake.csv"]
HEADERS = ["title", "text"];

# for fn in FNS:
#     for h in HEADERS:
#         o_fn = DIR + fn.split(".")[0] + "_" +  h + ".txt"
#         df = pd.read_csv(DIR + fn, usecols = [h])
#         df.to_csv(o_fn, index = False)

df_t = pd.read_csv(DIR + "true.csv", usecols = HEADERS)
df_f = pd.read_csv(DIR + "fake.csv", usecols = HEADERS)

for r in range(len(df_t)):
    o_fn = "input/" + str(r)
    with open(o_fn, "w") as f:
        for h in HEADERS:
            f.write(df_t.loc[r, h] + "\n")

for r in range(len(df_f)):
    o_fn = "input/" + str(r + len(df_t))
    if (r == 0):
        print("fake start: " + str(r + len(df_t)))
    with open(o_fn, "w") as f:
        for h in HEADERS:
            f.write(df_f.loc[r, h] + "\n")

        

