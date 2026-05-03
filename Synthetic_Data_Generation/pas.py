import warnings
import numpy as np
import pandas as pd


from pyapep.simsep import column



R = 8.3145


def lhs_sampling(bounds, n_samples=10, seed=10):
    rng = np.random.default_rng(seed)
    keys = list(bounds.keys())
    samples = np.zeros((n_samples, len(keys)))

    for j, key in enumerate(keys):
        low, high = bounds[key]
        cut = np.linspace(0, 1, n_samples + 1)
        u = rng.uniform(cut[:-1], cut[1:])
        rng.shuffle(u)
        samples[:, j] = low + u * (high - low)

    return pd.DataFrame(samples, columns=keys)


class MOFCO2VSA:
    def __init__(
        self,
        L=1.0,
        A_cros=0.01,
        N_node=31,
        T=313.15,
        PH=1.2,
        y_feed=(0.15, 0.85),
    ):
        self.L = L
        self.A = A_cros
        self.N = N_node
        self.T = T
        self.PH = PH
        self.y_feed = np.array(y_feed, dtype=float)

        self.eps = 0.40
        self.dp = 1.5e-3
        self.rho_s = 900.0

        self.M = [0.04401, 0.0280134]
        self.mu = [1.48e-5, 1.76e-5]

        self.D_ax = [1.0e-5, 1.0e-5]
        self.k_mtc = [1.0e-3, 8.0e-4]
        self.a_surf = 6.0 / self.dp

        self.dH = [35e3, 15e3]
        self.Cp_s = 900.0
        self.Cp_g = [37.1, 29.1]
        self.h_heat = 80.0

    def isotherm(self, P, T):
        Pco2 = np.asarray(P[0], dtype=float)
        Pn2 = np.asarray(P[1], dtype=float)
        T = np.asarray(T, dtype=float)

        # 임시 MOF 등온식 파라미터
        # 실제 MOF 데이터 fitting 값으로 교체 권장
        qsat_co2 = 4.0
        qsat_n2 = 0.8

        b0_co2 = 0.25
        b0_n2 = 0.015

        dH_co2 = 32e3
        dH_n2 = 12e3

        b_co2 = b0_co2 * np.exp(dH_co2 / R * (1.0 / T - 1.0 / 298.15))
        b_n2 = b0_n2 * np.exp(dH_n2 / R * (1.0 / T - 1.0 / 298.15))

        denom = 1.0 + b_co2 * Pco2 + b_n2 * Pn2

        q_co2 = qsat_co2 * b_co2 * Pco2 / denom
        q_n2 = qsat_n2 * b_n2 * Pn2 / denom

        return [q_co2, q_n2]

    def make_column(self, P_init, y_init):
        c = column(
            self.L,
            self.A,
            n_component=2,
            N_node=self.N,
        )

        c.adsorbent_info(
            self.isotherm,
            self.eps,
            self.dp,
            self.rho_s,
        )

        c.gas_prop_info(self.M, self.mu)

        c.mass_trans_info(
            self.k_mtc,
            self.a_surf,
            self.D_ax,
        )

        c.thermal_info(
            self.dH,
            self.Cp_s,
            self.Cp_g,
            self.h_heat,
        )

        P0 = np.ones(self.N) * P_init
        Tg0 = np.ones(self.N) * self.T
        Ts0 = np.ones(self.N) * self.T

        y0 = [
            np.ones(self.N) * y_init[0],
            np.ones(self.N) * y_init[1],
        ]

        q0 = self.isotherm(
            [y0[0] * P0, y0[1] * P0],
            Tg0,
        )

        c.initialC_info(P0, Tg0, Ts0, y0, q0)
        return c

    def set_boundary(self, c, Pin, Pout, y_in, v_superficial, forward=True):
        Q_in = max(v_superficial, 1e-9) * self.A * self.eps

        c.boundaryC_info(
            Pout,
            Pin,
            self.T,
            list(y_in),
            Cv_in=1e-3,
            Cv_out=1e-3,
            Q_inlet=Q_in,
            assigned_v_option=True,
            foward_flow_direction=forward,
        )

    def unpack_result(self, y_result):
        C_co2 = y_result[:, 0:self.N]
        C_n2 = y_result[:, self.N:2 * self.N]
        q_co2 = y_result[:, 2 * self.N:3 * self.N]
        q_n2 = y_result[:, 3 * self.N:4 * self.N]
        return C_co2, C_n2, q_co2, q_n2

    def update_initial_from_final(self, c, y_result, P_new):
        C_co2, C_n2, q_co2, q_n2 = self.unpack_result(y_result)

        C1 = np.maximum(C_co2[-1, :], 1e-12)
        C2 = np.maximum(C_n2[-1, :], 1e-12)
        Ctot = C1 + C2

        y1 = C1 / Ctot
        y2 = C2 / Ctot

        P0 = np.ones(self.N) * P_new
        Tg0 = np.ones(self.N) * self.T
        Ts0 = np.ones(self.N) * self.T

        q0 = [
            np.maximum(q_co2[-1, :], 0.0),
            np.maximum(q_n2[-1, :], 0.0),
        ]

        c.initialC_info(P0, Tg0, Ts0, [y1, y2], q0)

    def run_step(self, c, step_time, n_sec=10):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_result, z, t = c.run_ma(
                step_time,
                n_sec=n_sec,
                CPUtime_print=False,
            )

        return {
            "y_result": y_result,
            "z": z,
            "t": t,
        }

    def integrate_outlet_moles(self, step, v_superficial, outlet="left"):
        y_result = step["y_result"]
        t = step["t"]

        C_co2, C_n2, _, _ = self.unpack_result(y_result)

        idx = 0 if outlet == "left" else -1
        Q = max(v_superficial, 1e-9) * self.A * self.eps

        F_co2 = np.maximum(C_co2[:, idx], 0.0) * Q
        F_n2 = np.maximum(C_n2[:, idx], 0.0) * Q

        n_co2 = np.trapz(F_co2, t)
        n_n2 = np.trapz(F_n2, t)

        return n_co2, n_n2

    def simulate_one_cycle_with_column(
        self,
        c,
        tADS,
        PL,
        v0,
        t_press,
        t_depress,
        t_desorb=None,
        n_sec=10,
        calculate_metrics=True,
    ):
        if t_desorb is None:
            t_desorb = tADS

        history = {}

        # 1. Pressurization
        self.set_boundary(
            c,
            Pin=self.PH,
            Pout=self.PH,
            y_in=self.y_feed,
            v_superficial=max(v0 * 0.3, 1e-4),
            forward=True,
        )
        history["press"] = self.run_step(c, t_press, n_sec)
        self.update_initial_from_final(c, history["press"]["y_result"], self.PH)

        # 2. Adsorption
        self.set_boundary(
            c,
            Pin=self.PH,
            Pout=self.PH,
            y_in=self.y_feed,
            v_superficial=v0,
            forward=True,
        )
        history["ads"] = self.run_step(c, tADS, n_sec)
        self.update_initial_from_final(c, history["ads"]["y_result"], self.PH)

        # 3. Depressurization
        self.set_boundary(
            c,
            Pin=PL,
            Pout=PL,
            y_in=[0.01, 0.99],
            v_superficial=max(v0 * 0.15, 1e-4),
            forward=False,
        )
        history["depress"] = self.run_step(c, t_depress, n_sec)
        self.update_initial_from_final(c, history["depress"]["y_result"], PL)

        # 4. Vacuum desorption
        self.set_boundary(
            c,
            Pin=PL,
            Pout=PL,
            y_in=[0.01, 0.99],
            v_superficial=max(v0 * 0.15, 1e-4),
            forward=False,
        )
        history["desorb"] = self.run_step(c, t_desorb, n_sec)
        self.update_initial_from_final(c, history["desorb"]["y_result"], PL)

        if not calculate_metrics:
            return None, history

        C_feed = self.PH * 1e5 / (R * self.T)
        Q_feed = v0 * self.A * self.eps
        n_co2_feed = C_feed * Q_feed * self.y_feed[0] * tADS

        v_vac = max(v0 * 0.15, 1e-4)

        n_co2_dep, n_n2_dep = self.integrate_outlet_moles(
            history["depress"],
            v_superficial=v_vac,
            outlet="left",
        )

        n_co2_des, n_n2_des = self.integrate_outlet_moles(
            history["desorb"],
            v_superficial=v_vac,
            outlet="left",
        )

        n_co2_product = n_co2_dep + n_co2_des
        n_n2_product = n_n2_dep + n_n2_des

        total_product = n_co2_product + n_n2_product

        CO2_purity = n_co2_product / total_product if total_product > 0 else 0.0
        CO2_recovery = n_co2_product / n_co2_feed if n_co2_feed > 0 else 0.0

        metrics = {
            "CO2_purity": CO2_purity,
            "CO2_recovery": CO2_recovery,
            "n_CO2_feed": n_co2_feed,
            "n_CO2_product": n_co2_product,
            "n_N2_product": n_n2_product,
        }

        return metrics, history

    def simulate_10_cycles(
        self,
        tADS,
        PL,
        v0,
        t_press,
        t_depress,
        n_cycles=10,
        n_sec=10,
    ):
        c = self.make_column(
            P_init=PL,
            y_init=[0.01, 0.99],
        )

        final_metrics = None

        for cycle_idx in range(1, n_cycles + 1):
            calculate_metrics = cycle_idx == n_cycles

            final_metrics, _ = self.simulate_one_cycle_with_column(
                c=c,
                tADS=tADS,
                PL=PL,
                v0=v0,
                t_press=t_press,
                t_depress=t_depress,
                t_desorb=tADS,
                n_sec=n_sec,
                calculate_metrics=calculate_metrics,
            )

        return final_metrics


def generate_synthetic_dataset(
    n_samples=10,
    n_cycles=10,
    output_csv="mof_co2_vsa_lhs_10.csv",
    seed=10,
):
    bounds = {
        "tADS": (40.0, 160.0),
        "PL": (0.20, 0.50),
        "v0": (0.01, 0.05),
        "t_press": (15.0, 60.0),
        "t_depress": (15.0, 60.0),
    }

    lhs_df = lhs_sampling(bounds, n_samples=n_samples, seed=seed)

    model = MOFCO2VSA(
        L=1.0,
        A_cros=0.01,
        N_node=31,
        T=313.15,
        PH=1.2,
        y_feed=(0.15, 0.85),
    )

    rows = []

    for i, row in lhs_df.iterrows():
        params = row.to_dict()

        try:
            metrics = model.simulate_10_cycles(
                tADS=float(params["tADS"]),
                PL=float(params["PL"]),
                v0=float(params["v0"]),
                t_press=float(params["t_press"]),
                t_depress=float(params["t_depress"]),
                n_cycles=n_cycles,
                n_sec=10,
            )

            result = {
                "sample_id": i + 1,
                **params,
                **metrics,
                "n_cycles": n_cycles,
                "status": "ok",
            }

        except Exception as e:
            result = {
                "sample_id": i + 1,
                **params,
                "CO2_purity": np.nan,
                "CO2_recovery": np.nan,
                "n_CO2_feed": np.nan,
                "n_CO2_product": np.nan,
                "n_N2_product": np.nan,
                "n_cycles": n_cycles,
                "status": str(e),
            }

        rows.append(result)

        if (i + 1) % 10 == 0:
            print(f"{i + 1}/{n_samples} complete")

    result_df = pd.DataFrame(rows)
    result_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f"Saved: {output_csv}")
    return result_df


if __name__ == "__main__":
    df = generate_synthetic_dataset(
        n_samples=10,
        n_cycles=10,
        output_csv="mof_co2_vsa_lhs_10.csv",
        seed=10,
    )

    print(df.head())