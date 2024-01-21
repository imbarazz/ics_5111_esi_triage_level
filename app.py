# Gradio application to predict ESI triag level based on observation
# author: tbartolo
import math
import joblib
import gradio as gr

# START models
symptom_code_map = joblib.load('resources/etc/symptom_code_map.joblib')
age_min_max_scaler = joblib.load('resources/normalizers/AGE_min_max_scaler.joblib')
bpdias_min_max_scaler = joblib.load('resources/normalizers/BPDIAS_min_max_scaler.joblib')
bpsys_min_max_scaler = joblib.load('resources/normalizers/BPSYS_min_max_scaler.joblib')
lov_min_max_scaler = joblib.load('resources/normalizers/LOV_min_max_scaler.joblib')
painscale_min_max_scaler = joblib.load('resources/normalizers/PAINSCALE_min_max_scaler.joblib')
respr_min_max_scaler = joblib.load('resources/normalizers/RESPR_min_max_scaler.joblib')
tempf_discretizer_model = joblib.load('resources/normalizers/TEMPF_discretizer_model.joblib')
waittime_min_max_scaler = joblib.load('resources/normalizers/WAITTIME_min_max_scaler.joblib')
pulse_min_max_scaler = joblib.load('resources/normalizers/PULSE_min_max_scaler.joblib')
popct_min_max_scaler = joblib.load('resources/normalizers/POPCT_min_max_scaler.joblib')
gb_classifier_model = joblib.load('resources/algorithms/gb_classifier_model.joblib')
# END models

waittime_mean = 30.86
lov_mean = 236.02
age_mean = 37.71
tempf_mean = 94.95
pulse_mean = 89.86
respr_mean = 19.89
bpsys_mean = 119.74
bpdias_mean = 69.47
popct_mean = 94.52


def master_fn(waittime, lov, age, newborn, private_residnce, sex, arrems, ambtransfer, pulse, respr, bpsys, bpdias,
              popct, painscale, seen72, initial_visit, injury, injury72, etohab, alzhd, asthma, cancer, cebvd, ckd,
              copd, chf, cad, deprn, diabtyp1, diabtyp2, diabtyp0, esrd, hpe, edhiv, hyplipid, htn, obesity, osa,
              ostprsis, substab, racer, tempf, rfv1, rfv2, rfv3, rfv4, rfv5, trauma, overdose_poison, medical_surgical):
    def waittime_normalizer(value):
        if not value or value < 0:
            value = waittime_mean

        log10_value = math.log10(value)
        return waittime_min_max_scaler.transform([[log10_value]])[0][0]

    def lov_normalizer(value):
        if not value or value <= 0:
            value = lov_mean

        log10_value = math.log10(value)
        return lov_min_max_scaler.transform([[log10_value]])[0][0]

    def age_normalizer(value):
        if not value or value < 0:
            value = age_mean

        return age_min_max_scaler.transform([[value]])[0][0]

    def newborn_normalizer(value):
        if not value:
            value = "no"

        return yes_no_to_binary(value)

    def private_residnce_normalizer(value):
        if not value:
            value = "yes"

        return yes_no_to_binary(value)

    def sex_normalizer(value):
        if not value:
            value = "male"

        if value.lower() == "female":
            return 1
        elif value.lower() == "male":
            return 0
        else:
            raise ValueError("Illegal argument: SEX must \"female\" or \"male\"")

    def arrems_normalizer(value):
        if not value:
            value = "no"

        return yes_no_to_binary(value)

    def ambtransfer_normalizer(value):
        if not value:
            value = "no"

        return yes_no_to_binary(value)

    def pulse_normalizer(value):
        if not value or value <= 0:
            value = pulse_mean

        return pulse_min_max_scaler.transform([[value]])[0][0]

    def respr_normalizer(value):
        if not value or value <= 0:
            value = respr_mean

        if value > 35:
            value = 35  # clip

        return respr_min_max_scaler.transform([[value]])[0][0]

    def bpsys_normalizer(value):
        if not value or value <= 0:
            value = bpsys_mean

        return bpsys_min_max_scaler.transform([[value]])[0][0]

    def bpdias_normalizer(value):
        if not value or value <= 0:
            value = bpdias_mean

        return bpdias_min_max_scaler.transform([[value]])[0][0]

    def popct_normalizer(value):
        if not value or value <= 0:
            value = popct_mean

        log10_value = math.log10(value)
        return popct_min_max_scaler.transform([[log10_value]])[0][0]

    def painscale_normalizer(value):
        if not value:
            value = 0

        return painscale_min_max_scaler.transform([[value]])[0][0]

    def seen72_normalizer(value):
        if not value:
            value = "no"

        return yes_no_to_binary(value)

    def initial_visit_normalizer(value):
        if not value:
            value = "yes"

        return yes_no_to_binary(value)

    def injury_normalizer(value):
        if not value:
            value = "no"

        return yes_no_to_binary(value)

    def injury72_normalizer(value):
        if not value:
            value = "no"

        return yes_no_to_binary(value)

    def etohab_normalizer(value):
        if not value:
            value = "no"

        return yes_no_to_binary(value)

    def alzhd_normalizer(value):
        if not value:
            value = "no"

        return yes_no_to_binary(value)

    def asthma_normalizer(value):
        if not value:
            value = "no"

        return yes_no_to_binary(value)

    def cancer_normalizer(value):
        if not value:
            value = "no"

        return yes_no_to_binary(value)

    def cebvd_normalizer(value):
        if not value:
            value = "no"

        return yes_no_to_binary(value)

    def ckd_normalizer(value):
        if not value:
            value = "no"

        return yes_no_to_binary(value)

    def copd_normalizer(value):
        if not value:
            value = "no"

        return yes_no_to_binary(value)

    def chf_normalizer(value):
        if not value:
            value = "no"

        return yes_no_to_binary(value)

    def cad_normalizer(value):
        if not value:
            value = "no"

        return yes_no_to_binary(value)

    def deprn_normalizer(value):
        if not value:
            value = "no"

        return yes_no_to_binary(value)

    def diabtyp2_normalizer(value):
        if not value:
            value = "no"

        return yes_no_to_binary(value)

    def diabtyp1_normalizer(value):
        if not value:
            value = "no"

        return yes_no_to_binary(value)

    def diabtyp0_normalizer(value):
        if not value:
            value = "no"

        return yes_no_to_binary(value)

    def esrd_normalizer(value):
        if not value:
            value = "no"

        return yes_no_to_binary(value)

    def hpe_normalizer(value):
        if not value:
            value = "no"

        return yes_no_to_binary(value)

    def edhiv_normalizer(value):
        if not value:
            value = "no"

        return yes_no_to_binary(value)

    def hyplipid_normalizer(value):
        if not value:
            value = "no"

        return yes_no_to_binary(value)

    def htn_normalizer(value):
        if not value:
            value = "no"

        return yes_no_to_binary(value)

    def obesity_normalizer(value):
        if not value:
            value = "no"

        return yes_no_to_binary(value)

    def osa_normalizer(value):
        if not value:
            value = "no"

        return yes_no_to_binary(value)

    def ostprsis_normalizer(value):
        if not value:
            value = "no"

        return yes_no_to_binary(value)

    def substab_normalizer(value):
        if not value:
            value = "no"

        return yes_no_to_binary(value)

    def racer_normalizer(value):
        if not value:
            value = "other"

        if value.lower() == "white":
            return [0, 1]
        elif value.lower() == "black":
            return [1, 0]
        elif value.lower() == "other":
            return [0, 0]
        else:
            raise ValueError("Illegal argument: RACER must \"white\" or \"black\" or \"other\"")

    def tempf_normalizer(value):
        if not value or value < 70:
            value = tempf_mean

        return tempf_discretizer_model.transform([[value * 10]]).flatten()

    def rfv1_normalizer(value):
        if not value:
            value = 0

        return rfv_to_mapping(value)

    def rfv2_normalizer(value):
        if not value:
            value = 0

        return rfv_to_mapping(value)

    def rfv3_normalizer(value):
        if not value:
            value = 0

        return rfv_to_mapping(value)

    def rfv4_normalizer(value):
        if not value:
            value = 0

        return rfv_to_mapping(value)

    def rfv5_normalizer(value):
        if not value:
            value = 0

        return rfv_to_mapping(value)

    def trauma_normalizer(value):
        if not value:
            value = "no"

        return yes_no_to_binary(value)

    def overdose_poison_normalizer(value):
        if not value:
            value = "no"

        return yes_no_to_binary(value)

    def medical_surgical_normalizer(value):
        if not value:
            value = "no"

        return yes_no_to_binary(value)

    def yes_no_to_binary(value):
        if value.lower() == "yes":
            return 1
        elif value.lower() == "no":
            return 0
        else:
            raise ValueError("Illegal argument: Value must \"yes\" or \"no\"")

    def rfv_to_mapping(value):
        value = symptom_code_map.get(value * 10)

        if value or value == 0:
            return decimal_to_binary_array(value, 13)
        else:
            raise ValueError("Illegal argument: Value must be valid symptom code")

    def decimal_to_binary_array(number, array_size):
        binary_representation = bin(number)[2:]  # Convert to binary and remove '0b' prefix
        binary_array = [int(bit) for bit in binary_representation.zfill(array_size)[-array_size:]]
        return binary_array

    observation = []

    observation.append(waittime_normalizer(waittime))
    observation.append(lov_normalizer(lov))
    observation.append(age_normalizer(age))
    observation.append(newborn_normalizer(newborn))
    observation.append(private_residnce_normalizer(private_residnce))
    observation.append(sex_normalizer(sex))
    observation.append((arrems_normalizer(arrems)))
    observation.append((ambtransfer_normalizer(ambtransfer)))
    observation.append(pulse_normalizer(pulse))
    observation.append(respr_normalizer(respr))
    observation.append(bpsys_normalizer(bpsys))
    observation.append(bpdias_normalizer(bpdias))
    observation.append(popct_normalizer(popct))
    observation.append(painscale_normalizer(painscale))
    observation.append(seen72_normalizer(seen72))
    observation.append(initial_visit_normalizer(initial_visit))
    observation.append(injury_normalizer(injury))
    observation.append(injury72_normalizer(injury72))
    observation.append(etohab_normalizer(etohab))
    observation.append(alzhd_normalizer(alzhd))
    observation.append(asthma_normalizer(asthma))
    observation.append(cancer_normalizer(cancer))
    observation.append(cebvd_normalizer(cebvd))
    observation.append(ckd_normalizer(ckd))
    observation.append(copd_normalizer(copd))
    observation.append(chf_normalizer(chf))
    observation.append(cad_normalizer(cad))
    observation.append(deprn_normalizer(deprn))
    observation.append(diabtyp1_normalizer(diabtyp1))
    observation.append(diabtyp2_normalizer(diabtyp2))
    observation.append(diabtyp0_normalizer(diabtyp0))
    observation.append(esrd_normalizer(esrd))
    observation.append(hpe_normalizer(hpe))
    observation.append(edhiv_normalizer(edhiv))
    observation.append(hyplipid_normalizer(hyplipid))
    observation.append(htn_normalizer(htn))
    observation.append(obesity_normalizer(obesity))
    observation.append(osa_normalizer(osa))
    observation.append(ostprsis_normalizer(ostprsis))
    observation.append(substab_normalizer(substab))

    observation = (observation + racer_normalizer(racer) + tempf_normalizer(tempf).tolist() + rfv1_normalizer(rfv1)
                   + rfv2_normalizer(rfv2)) + rfv3_normalizer(rfv3) + rfv4_normalizer(rfv4) + rfv5_normalizer(rfv5)

    observation.append(trauma_normalizer(trauma))
    observation.append(overdose_poison_normalizer(overdose_poison))
    observation.append(medical_surgical_normalizer(medical_surgical))

    return gb_classifier_model.predict([observation])[0]


demo = gr.Interface(
    master_fn,
    [
        "number",  # waittime
        "number",  # lov
        "number",  # age
        gr.Radio(["yes", "no"]),  # newborn
        gr.Radio(["yes", "no"]),  # private_residnce
        gr.Radio(["male", "female"]),  # sex
        gr.Radio(["yes", "no"]),  # arrems
        gr.Radio(["yes", "no"]),  # ambtransfer
        "number",  # pulse
        "number",  # respr
        "number",  # bpsys
        "number",  # bpdias
        "number",  # popct
        "number",  # painscale
        gr.Radio(["yes", "no"]),  # seen72
        gr.Radio(["yes", "no"]),  # initial_visit
        gr.Radio(["yes", "no"]),  # injury
        gr.Radio(["yes", "no"]),  # injury72
        gr.Radio(["yes", "no"]),  # etohab
        gr.Radio(["yes", "no"]),  # alzhd
        gr.Radio(["yes", "no"]),  # asthma
        gr.Radio(["yes", "no"]),  # cancer
        gr.Radio(["yes", "no"]),  # cebvd
        gr.Radio(["yes", "no"]),  # ckd
        gr.Radio(["yes", "no"]),  # copd
        gr.Radio(["yes", "no"]),  # chf
        gr.Radio(["yes", "no"]),  # cad
        gr.Radio(["yes", "no"]),  # deprn
        gr.Radio(["yes", "no"]),  # diabtyp1
        gr.Radio(["yes", "no"]),  # diabtyp2
        gr.Radio(["yes", "no"]),  # diabtyp0
        gr.Radio(["yes", "no"]),  # esrd
        gr.Radio(["yes", "no"]),  # hpe
        gr.Radio(["yes", "no"]),  # edhiv
        gr.Radio(["yes", "no"]),  # hyplipid
        gr.Radio(["yes", "no"]),  # htn
        gr.Radio(["yes", "no"]),  # obesity
        gr.Radio(["yes", "no"]),  # osa
        gr.Radio(["yes", "no"]),  # ostprsis
        gr.Radio(["yes", "no"]),  # substab
        gr.Radio(["white", "black", "other"]),  # racer
        "number",  # tempf
        "number",  # rfv1
        "number",  # rfv2
        "number",  # rfv3
        "number",  # rfv4
        "number",  # rfv5
        gr.Radio(["yes", "no"]),  # trauma
        gr.Radio(["yes", "no"]),  # overdose_poison
        gr.Radio(["yes", "no"])  # medical_surgical
    ],
    "number",
    title="ESI Triage Level Predictor",
    description="Enter the patient attributes and symptoms to perform the prediction. Note this application is a POC "
                "and by no means claims any sort of accuracy in its predictions.",
)

if __name__ == "__main__":
    demo.launch(show_api=False)
