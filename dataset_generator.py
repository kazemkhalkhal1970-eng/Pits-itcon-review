import numpy as np, json
from pathlib import Path
np.random.seed(42)
SOILS={"dry_sand":{"k_c":0.99,"k_phi":1528.43,"n":1.0,"c":1.04,"phi":28.0,"rho":1590,"label":0},"wet_sand":{"k_c":4.45,"k_phi":1198.93,"n":1.0,"c":5.52,"phi":36.0,"rho":1810,"label":1},"clay_loam":{"k_c":13.32,"k_phi":1340.46,"n":0.6,"c":11.65,"phi":19.5,"rho":1540,"label":2},"silty_loam":{"k_c":5.95,"k_phi":1528.43,"n":0.7,"c":7.45,"phi":22.5,"rho":1420,"label":3},"gravel":{"k_c":1.40,"k_phi":3117.61,"n":0.95,"c":2.45,"phi":35.0,"rho":2010,"label":4},"wet_clay":{"k_c":31.00,"k_phi":1013.50,"n":0.5,"c":18.95,"phi":14.0,"rho":1820,"label":5}}
EXC={"L1":6.20,"L2":3.30,"L3":1.55,"m_bucket":1850.0,"V_max":1.42,"p_hydr_nominal":34.3e6,"b":0.45,"piston_area":0.0045,"eta":0.92,"flow_rate":0.0008}
PRIMS=[{"name":"reach","t":2.5,"label":0},{"name":"penetrate","t":2.5,"label":1},{"name":"curl","t":2.5,"label":2},{"name":"lift","t":2.5,"label":3},{"name":"swing","t":3.5,"label":4},{"name":"dump","t":2.5,"label":5},{"name":"return","t":4.0,"label":6}]
NC=len(PRIMS);FS=30;P_NOM=EXC["p_hydr_nominal"];b=EXC["b"]
def beker_p(z,s):return(s["k_c"]/b+s["k_phi"])*(z**s["n"])
def mc(s,sig):return s["c"]*1000.0+sig*np.tan(np.deg2rad(s["phi"]))
def hp(t,a=0.0045,e=0.92):return t/(a*e)+P_NOM*0.55
def bf(z,s,v_pen=0.15):
    if z<=0:return 0.0
    p=beker_p(z,s);F_n=p*(b*0.35);tau=mc(s,p);F_s=tau*(2*b*0.20);return F_n+F_s+850.0*v_pen
def gen(soil,idx,op=0.5):
    durs=np.array([p["t"] for p in PRIMS])*(0.85+0.3*op);durs=durs*(1.0+0.06*np.random.randn(len(durs)));durs=np.clip(durs,0.8,6.0)
    tot=float(durs.sum());n=max(int(tot*FS),80)
    ja=np.zeros((n,4));hpr=np.zeros((n,4));bd=np.zeros(n);lb=np.zeros(n,dtype=np.int32)
    rest=np.array([-0.55,1.55,-1.10,-0.20]);mp=0.85*(1.0-0.4*(soil["k_phi"]/3117.61));mp=np.clip(mp,0.20,0.85)
    fl=np.clip(0.65+0.15*(soil["c"]/20.0),0.55,0.92)
    cum=np.concatenate([[0],np.cumsum(durs)]);bnd=(cum*FS).astype(int);bnd[-1]=n;hpr[:,:]=P_NOM*0.55
    for i,prim in enumerate(PRIMS):
        sa,ee=bnd[i],bnd[i+1];m=ee-sa
        if m<=0:continue
        lb[sa:ee]=prim["label"];tl=np.linspace(0,durs[i],m);nm=prim["name"]
        if nm=="reach":
            ja[sa:ee,0]=np.linspace(rest[0],-0.75,m)+0.02*np.sin(tl*2);ja[sa:ee,1]=np.linspace(rest[1],1.90,m)+0.01*np.sin(tl*3);ja[sa:ee,2]=rest[2];ja[sa:ee,3]=rest[3]
            for k in range(m):hpr[sa+k,0]=P_NOM*0.58+0.02*np.sin(tl[k]*3)*P_NOM;hpr[sa+k,1]=P_NOM*0.56;hpr[sa+k,2]=P_NOM*0.55;hpr[sa+k,3]=P_NOM*0.57
        elif nm=="penetrate":
            dp=mp*(1-np.exp(-3*tl/durs[i]))+0.015*np.sin(tl*6)*tl/durs[i];dp=np.clip(dp,0,mp);bd[sa:ee]=dp
            ja[sa:ee,0]=-0.75+0.04*tl/durs[i];ja[sa:ee,1]=1.90-0.05*tl/durs[i];ja[sa:ee,2]=-1.10+0.20*tl/durs[i];ja[sa:ee,3]=rest[3]
            for k in range(m):F=bf(dp[k],soil,v_pen=mp/durs[i]);hpr[sa+k,0]=hp(F*EXC["L1"]*0.4);hpr[sa+k,1]=hp(F*EXC["L2"]);hpr[sa+k,2]=hp(F*EXC["L3"]*0.6);hpr[sa+k,3]=P_NOM*0.58
        elif nm=="curl":
            ja[sa:ee,2]=-0.90+0.85*(tl/durs[i]);ja[sa:ee,1]=1.85-0.10*tl/durs[i];ja[sa:ee,0]=-0.71+0.05*tl/durs[i];ja[sa:ee,3]=rest[3];bd[sa:ee]=mp*(1-tl/durs[i])*0.4
            for k in range(m):F=bf(bd[sa+k],soil,v_pen=0.05)*1.4;hpr[sa+k,2]=hp(F*EXC["L3"]*0.9);hpr[sa+k,0]=P_NOM*0.60;hpr[sa+k,1]=P_NOM*0.58;hpr[sa+k,3]=P_NOM*0.59
        elif nm=="lift":
            dp=np.clip(mp*0.4*(1-tl/durs[i]),0,mp);bd[sa:ee]=dp;ja[sa:ee,0]=-0.66+0.45*tl/durs[i];ja[sa:ee,1]=1.75-0.35*tl/durs[i];ja[sa:ee,2]=-0.05+0.20*tl/durs[i];ja[sa:ee,3]=rest[3]
            for k in range(m):ml=EXC["m_bucket"]+1800.0*fl;F=ml*9.81+bf(dp[sa+k-sa],soil,v_pen=0.02)*0.3;hpr[sa+k,0]=hp(F*EXC["L1"]*0.55);hpr[sa+k,1]=hp(F*EXC["L2"]*0.35);hpr[sa+k,2]=P_NOM*0.59;hpr[sa+k,3]=P_NOM*0.58
        elif nm=="swing":
            sw=np.sin(0.5*np.pi*tl/durs[i])*1.40;ja[sa:ee,3]=rest[3]+sw;ja[sa:ee,0]=-0.21;ja[sa:ee,1]=1.40;ja[sa:ee,2]=0.15;ml=EXC["m_bucket"]+1800.0*fl;sw_acc=np.gradient(np.gradient(sw,1.0/FS))
            for k in range(m):ts=ml*4.0*sw_acc[k]*1.2;hpr[sa+k,0]=P_NOM*0.60;hpr[sa+k,1]=P_NOM*0.58;hpr[sa+k,2]=P_NOM*0.59;hpr[sa+k,3]=hp(abs(ts))
        elif nm=="dump":
            ja[sa:ee,2]=0.15+0.65*np.sin(0.5*np.pi*tl/durs[i]);ja[sa:ee,0]=-0.21-0.10*np.sin(np.pi*tl/durs[i]);ja[sa:ee,1]=1.40+0.20*np.sin(np.pi*tl/durs[i]);ja[sa:ee,3]=rest[3]+1.40
            for k in range(m):hpr[sa+k,0]=P_NOM*0.60;hpr[sa+k,1]=P_NOM*0.58;hpr[sa+k,2]=hp(EXC["m_bucket"]*9.81*0.5);hpr[sa+k,3]=P_NOM*0.59
        elif nm=="return":
            sw=1.40*(1-tl/durs[i]);ja[sa:ee,3]=rest[3]+sw;ja[sa:ee,0]=np.linspace(-0.31,rest[0],m);ja[sa:ee,1]=np.linspace(1.60,rest[1],m);ja[sa:ee,2]=np.linspace(0.80,rest[2],m)
            for k in range(m):hpr[sa+k,0]=P_NOM*0.57+0.01*np.sin(tl[k])*P_NOM;hpr[sa+k,1]=P_NOM*0.56;hpr[sa+k,2]=P_NOM*0.55;hpr[sa+k,3]=P_NOM*0.58+0.02*np.sin(tl[k]*2)*P_NOM
    ja+=0.004*np.random.randn(*ja.shape);hpr+=0.012*P_NOM*np.random.randn(*hpr.shape);hpr=np.clip(hpr,P_NOM*0.5,P_NOM*1.8)
    pm=(lb==1)
    if pm.sum()>0:zi=np.trapezoid(bd[pm],dx=1.0/FS);V=EXC["V_max"]*fl*(zi/(mp*durs[1]));V=np.clip(V,0,EXC["V_max"])
    else:V=0.0
    E=np.sum(hpr*EXC["flow_rate"]/FS);sf=np.array([soil["k_c"],soil["k_phi"],soil["n"],soil["c"],soil["phi"],soil["rho"]])
    return{"ja":ja.astype(np.float32),"hp":hpr.astype(np.float32),"sf":sf.astype(np.float32),"sl":int(soil["label"]),"lb":lb.astype(np.int32),"bd":bd.astype(np.float32),"meta":{"soil_name":[k for k,v in SOILS.items() if v["label"]==soil["label"]][0],"volume_m3":float(V),"energy_J":float(E),"max_pen_m":float(mp),"duration_s":float(tot),"operator_style":float(op),"cycle_idx":int(idx),"n_frames":int(n)}}
def main():
    out=Path("/home/z/my-project/download/dataset");out.mkdir(parents=True,exist_ok=True);all_c=[]
    for sn,soil in SOILS.items():
        for i in range(200):ci=soil["label"]*200+i;op=np.clip(0.3+0.5*np.random.rand(),0.2,0.95);c=gen(soil,ci,op);c["meta"]["soil_name"]=sn;all_c.append(c)
    tr,te=[],[]
    for sn in SOILS:sub=[c for c in all_c if c["meta"]["soil_name"]==sn];tr.extend(sub[:160]);te.extend(sub[160:])
    def stack(cs):
        ml=max(c["ja"].shape[0] for c in cs);n=len(cs)
        Xj=np.zeros((n,ml,4),dtype=np.float32);Xp=np.zeros((n,ml,4),dtype=np.float32);Y=np.full((n,ml),-1,dtype=np.int32);SF=np.zeros((n,6),dtype=np.float32);SL=np.zeros((n,),dtype=np.int32);M=np.zeros((n,ml),dtype=np.float32);mt=[]
        for i,c in enumerate(cs):L=c["ja"].shape[0];Xj[i,:L]=c["ja"];Xp[i,:L]=c["hp"];Y[i,:L]=c["lb"];SF[i]=c["sf"];SL[i]=c["sl"];M[i,:L]=1.0;mt.append(c["meta"])
        return Xj,Xp,Y,SF,SL,M,mt,ml
    Xjt,Xpt,Yt,SFt,SLt,Mt,mtt,ml1=stack(tr);Xje,Xpe,Ye,SFe,SLe,Me,mte,ml2=stack(te);ml=max(ml1,ml2)
    def pad(X,t):
        if X.shape[1]>=t:return X
        p=np.zeros((X.shape[0],t-X.shape[1],X.shape[2]),dtype=X.dtype);return np.concatenate([X,p],axis=1)
    def pad1(X,t,v):
        if X.shape[1]>=t:return X
        p=np.full((X.shape[0],t-X.shape[1]),v,dtype=X.dtype);return np.concatenate([X,p],axis=1)
    Xjt=pad(Xjt,ml);Xpt=pad(Xpt,ml);Yt=pad1(Yt,ml,-1);Mt=pad1(Mt,ml,0);Xje=pad(Xje,ml);Xpe=pad(Xpe,ml);Ye=pad1(Ye,ml,-1);Me=pad1(Me,ml,0)
    np.savez_compressed(out/"excavator_action_dataset.npz",X_joints_train=Xjt,X_press_train=Xpt,Y_train=Yt,soil_features_train=SFt,soil_labels_train=SLt,mask_train=Mt,X_joints_test=Xje,X_press_test=Xpe,Y_test=Ye,soil_features_test=SFe,soil_labels_test=SLe,mask_test=Me,num_classes=NC,fs=FS,max_len=ml,plate_width=b,piston_area=EXC["piston_area"],eta=EXC["eta"],flow_rate=EXC["flow_rate"])
    with open(out/"metadata.json","w") as f:json.dump({"n_train":len(tr),"n_test":len(te),"n_soils":6,"n_classes":NC,"fs":FS,"max_len":int(ml),"classes":[p["name"] for p in PRIMS],"soils":{k:{kk:float(vv) for kk,vv in v.items()} for k,v in SOILS.items()},"excavator":{k:float(v) for k,v in EXC.items()}},f,indent=2)
    with open(out/"cycle_meta_train.json","w") as f:json.dump(mtt,f,indent=2)
    with open(out/"cycle_meta_test.json","w") as f:json.dump(mte,f,indent=2)
    print(f"Done: train={len(tr)} test={len(te)} max_len={ml}")
if __name__=="__main__":main()
