import streamlit as st
st.set_page_config(page_title="Exoplanet Mission Planner", layout="wide")
import pandas as pd
import io
import requests
import matplotlib.pyplot as plt
import numpy as np


# page setup

st.set_page_config(
    page_title="Exoplanet Mission Planner",
    layout="wide"
)

@st.cache_data(show_spinner=False)
def fetch_exoplanet_data():
    """Fetch live data from NASA Exoplanet Archive."""
    base_url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    query = (
        "select pl_name, st_teff, st_rad, pl_orbsmax, pl_eqt, pl_rade, sy_dist "
        "from ps "
        "where default_flag = 1 and pl_orbsmax is not null and st_teff is not null and sy_dist is not null"
    )
    url = f"{base_url}?query={query}&format=csv"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return pd.read_csv(io.StringIO(response.text))
    except Exception as e:
        st.error(f"Failed to fetch NASA data: {e}")
        return None

def calculate_teq(t_star, r_star, a, albedo=0.3):
    """Stefan-Boltzmann equilibrium temperature."""
    r_star_au = r_star / 215.03
    return t_star * ((r_star_au / (2 * a)) ** 0.5) * ((1 - albedo) ** 0.25)

def parsecs_to_meters(pc):
    return pc * 3.085677581e16

def seconds_to_years(s):
    return s / (365.25 * 24 * 3600)

#propulsion system
class PropulsionSystem:
    def __init__(self, name, exhaust_velocity_ms, accel_ms2, efficiency=0.5):
        self.name = name
        self.ve = exhaust_velocity_ms
        self.accel = accel_ms2
        self.efficiency = efficiency
    
    def calculate_mission(self, distance_pc, num_people, mass_per_person=1000):
        d_m = parsecs_to_meters(distance_pc)
        d_half = d_m / 2
        t_half = np.sqrt(2 * d_half / self.accel)
        total_time_y = seconds_to_years(2 * t_half)
        v_max = self.accel * t_half
        c = 299792458
        frac_c = v_max / c
        delta_v = 2 * v_max
        payload_mass = num_people * mass_per_person
        mass_ratio = np.exp(delta_v / self.ve)
        fuel_mass = payload_mass * (mass_ratio - 1)
        avg_mass = payload_mass + fuel_mass / 2
        thrust = avg_mass * self.accel
        power = thrust * self.ve / (2 * self.efficiency)
        
        return {
            'propulsion': self.name,
            'distance_pc': distance_pc,
            'distance_ly': distance_pc * 3.26156,
            'total_time_y': total_time_y,
            'max_velocity_ms': v_max,
            'fraction_of_c': frac_c,
            'delta_v_ms': delta_v,
            'payload_mass_kg': payload_mass,
            'fuel_mass_kg': fuel_mass,
            'total_mass_kg': payload_mass + fuel_mass,
            'mass_ratio': mass_ratio,
            'thrust_n': thrust,
            'power_mw': power / 1e6
        }

PROPULSION_CATALOG = [
    PropulsionSystem("Current Ion Drive (NEXT) 40000 m/s", 40000, 0.001, 0.6),
    PropulsionSystem("Advanced Ion Drive (VASIMR) 50000 m/s", 50000, 0.01, 0.5),
    PropulsionSystem("Nuclear Pulse (Orion) 30000000 m/s", 30000000, 10.0, 0.05),
    PropulsionSystem("Fusion Drive (Project Daedalus) 10000000 m/s", 10000000, 0.1, 0.3),
    PropulsionSystem("Antimatter Catalyzed 10000000 m/s", 10000000, 1.0, 0.4),
    PropulsionSystem("Photonic Laser Sail 299792458 m/s", 299792458, 0.01, 0.9),
]

#visualisation
def create_visualization(df, hz_planets, prime_candidates):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    ax1.scatter(df['pl_orbsmax'], df['calc_teq'], c='gray', alpha=0.3, s=20, label='All Planets')
    ax1.scatter(hz_planets['pl_orbsmax'], hz_planets['calc_teq'], c='blue', alpha=0.6, s=40, label='Habitable Zone')
    ax1.scatter(prime_candidates['pl_orbsmax'], prime_candidates['calc_teq'], 
                c='gold', edgecolors='black', s=100, linewidths=1.5, label='Prime Rocky', zorder=5)
    
    ax1.axhline(y=273, color='red', linestyle='--', alpha=0.5, label='Water Freezing')
    ax1.axhline(y=373, color='red', linestyle='--', alpha=0.5, label='Water Boiling')
    ax1.set_xscale('log')
    ax1.set_title('Equilibrium Temperature vs. Orbital Distance')
    ax1.set_xlabel('Semi-major Axis (AU) - Log Scale')
    ax1.set_ylabel('Temperature (K)')
    ax1.legend()
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    
    if len(prime_candidates) > 0:
        distances = prime_candidates['sy_dist'].dropna()
        ax2.hist(distances, bins=20, color='gold', edgecolor='black', alpha=0.7)
        ax2.axvline(distances.median(), color='red', linestyle='--', linewidth=2, 
                    label=f'Median: {distances.median():.1f} pc')
        ax2.set_title('Distance Distribution of Prime Candidates')
        ax2.set_xlabel('Distance from Earth (parsecs)')
        ax2.set_ylabel('Number of Planets')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

#website
st.title("Exoplanet Mission Planner")
st.markdown("Fetch live data from NASA, identify habitable worlds, and calculate interstellar missions.")

# controls
with st.sidebar:
    st.header("Mission Settings")
    people = st.number_input("Passengers", min_value=1, max_value=10000, value=10)
    mass_pp = st.number_input("Mass per person (kg)", min_value=100, value=1000, step=100)
    
    prop_names = [p.name for p in PROPULSION_CATALOG]
    selected_prop = st.selectbox("Propulsion System", prop_names)
    propulsion = next(p for p in PROPULSION_CATALOG if p.name == selected_prop)
    
    st.markdown("---")
    st.caption("Uses Tsiolkovsky rocket equation & constant acceleration profile.")

# content
if st.button("Fetch Exoplanet Data from NASA", type="primary"):
    with st.spinner("Connecting to NASA Exoplanet Archive..."):
        df = fetch_exoplanet_data()
        if df is not None:
            st.session_state.df = df
            st.success(f"Loaded {len(df)} confirmed exoplanets!")

if 'df' in st.session_state:
    df = st.session_state.df.copy()
    df['calc_teq'] = df.apply(
        lambda x: calculate_teq(x['st_teff'], x['st_rad'], x['pl_orbsmax']), axis=1
    )
    
    hz_mask = (df['calc_teq'] >= 273) & (df['calc_teq'] <= 373)
    rocky_mask = (df['pl_rade'] <= 1.6)
    hz_planets = df[hz_mask].copy()
    prime_candidates = df[hz_mask & rocky_mask].copy()
    
    metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🪐 Total Planets", len(df))
    c2.metric("🌡️ Habitable Zone", len(hz_planets))
    c3.metric("🌍 Prime Rocky", len(prime_candidates))
    if len(prime_candidates) > 0:
        c4.metric("📏 Avg Distance", f"{prime_candidates['sy_dist'].mean():.1f} pc")
    #charts
    st.subheader("Data Visualization")
    fig = create_visualization(df, hz_planets, prime_candidates)
    st.pyplot(fig)
    
    #selection and planning
    if len(prime_candidates) > 0:
        st.subheader("Select Target Planet")
        
        display_df = prime_candidates[['pl_name', 'calc_teq', 'pl_rade', 'sy_dist', 'pl_orbsmax']].sort_values('calc_teq')
        display_df.columns = ['Name', 'Temp (K)', 'Radius (Earth)', 'Distance (pc)', 'Orbit (AU)']
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        planet_names = display_df['Name'].tolist()
        selected_name = st.selectbox("Choose destination", planet_names)
        selected = prime_candidates[prime_candidates['pl_name'] == selected_name].iloc[0]
        
        if st.button("Calculate Mission", type="primary"):
            with st.spinner("Computing trajectory..."):
                results = propulsion.calculate_mission(selected['sy_dist'], people, mass_pp)
                
                st.subheader(f"Mission Report: {selected_name}")
                
                #metrics
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Travel Time", f"{results['total_time_y']:,.0f} years")
                m2.metric("Max Velocity", f"{results['fraction_of_c']:.4f}c")
                m3.metric("Fuel Mass", f"{results['fuel_mass_kg']:.2e} kg")
                m4.metric("Power Needed", f"{results['power_mw']:,.0f} MW")
                
                #physics
                with st.expander("Physics Report"):
                    st.markdown(f"""
                    - **Propulsion:** {results['propulsion']}
                    - **Distance:** {results['distance_pc']:.2f} pc ({results['distance_ly']:.2f} ly)
                    - **Delta-V:** {results['delta_v_ms']:,.0f} m/s
                    - **Mass Ratio:** {results['mass_ratio']:.2f}
                    - **Thrust:** {results['thrust_n']:,.2f} N
                    """)
                
                #check
                st.subheader("Feasibility Assessment")
                checks = []
                if results['total_time_y'] > 10000:
                    checks.append("❌ Duration exceeds civilization timescales")
                elif results['total_time_y'] > 1000:
                    checks.append("⚠️ Multi-generational mission required")
                elif results['total_time_y'] > 100:
                    checks.append("⚠️ Long-duration mission")
                else:
                    checks.append("✅ Duration within human lifetime")
                
                if results['fraction_of_c'] > 0.5:
                    checks.append("⚠️ Relativistic speeds require exotic propulsion")
                elif results['fuel_mass_kg'] > 1e15:
                    checks.append("❌ Fuel exceeds planetary mass")
                elif results['fuel_mass_kg'] > 1e9:
                    checks.append("⚠️ Fuel requirements massive")
                else:
                    checks.append("✅ Fuel requirements manageable")
                
                if results['power_mw'] > 1e6:
                    checks.append("❌ Power exceeds global civilization output")
                elif results['power_mw'] > 1e3:
                    checks.append("⚠️ Requires dedicated reactor infrastructure")
                else:
                    checks.append("✅ Power requirements feasible")
                
                for check in checks:
                    st.write(check)
                
                #comparison table
                with st.expander("🔍 Compare All Propulsion Systems"):
                    comp_data = []
                    for prop in PROPULSION_CATALOG:
                        try:
                            r = prop.calculate_mission(selected['sy_dist'], people, mass_pp)
                            comp_data.append({
                                'System': prop.name,
                                'Time (years)': f"{r['total_time_y']:,.1f}",
                                'Fuel (kg)': f"{r['fuel_mass_kg']:.2e}" if r['fuel_mass_kg'] < 1e15 else "Impossible",
                                'Max v/c': f"{r['fraction_of_c']:.6f}",
                                'Power (MW)': f"{r['power_mw']:,.1f}"
                            })
                        except:
                            comp_data.append({
                                'System': prop.name,
                                'Time (years)': 'N/A',
                                'Fuel (kg)': 'Impossible',
                                'Max v/c': 'N/A',
                                'Power (MW)': 'N/A'
                            })
                    st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)
    else:
        st.warning("No prime rocky candidates found in the habitable zone.")
else:
    st.info("Click **Fetch Exoplanet Data** to load data from NASA.")