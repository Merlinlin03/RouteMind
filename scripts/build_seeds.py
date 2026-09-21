"""Rebuild hand-authored synthetic seed annotations with explicit provenance."""
from pathlib import Path

from routemind.data import write_jsonl
from routemind.schema import DatasetRecord, validate_evidence

LANGUAGES = ("en", "es", "pt", "id", "hi", "ar")
CLAIMS = ["offline use", "uso sin conexión", "uso offline", "penggunaan offline", "ऑफलाइन उपयोग", "الاستخدام دون اتصال"]
EXPERIENCES = ["requires internet for everything", "necesita internet para todo", "exige internet para tudo",
               "membutuhkan internet untuk semuanya", "हर काम के लिए इंटरनेट चाहिए", "يحتاج الإنترنت لكل شيء"]
INITIAL_BILLING = ["I was charged twice.", "Me cobraron dos veces.", "Fui cobrado duas vezes.",
                   "Saya ditagih dua kali.", "मुझसे दो बार पैसे काटे गए।", "تم الخصم مني مرتين."]
ASSISTANT_BILLING = ["Please check whether the duplicate charge is still present.",
                     "Comprueba si el cobro duplicado sigue apareciendo.",
                     "Verifique se a cobrança duplicada ainda aparece.",
                     "Periksa apakah tagihan ganda masih ada.",
                     "कृपया जाँचें कि दोहरी कटौती अभी भी दिखाई दे रही है या नहीं।",
                     "يرجى التحقق مما إذا كان الخصم المكرر لا يزال ظاهراً."]
# One semantic scenario per group. Translations stay together during splitting.
TEXTS = {
    "premium": [
        "I paid through Stripe, but Premium is still locked.",
        "Pagué con Stripe, pero Premium sigue bloqueado.",
        "Paguei pelo Stripe, mas o Premium continua bloqueado.",
        "Saya sudah membayar lewat Stripe, tetapi Premium masih terkunci.",
        "मैंने Stripe से भुगतान किया, लेकिन Premium अभी भी बंद है।",
        "دفعت عبر Stripe، لكن Premium لا يزال مقفلاً.",
    ],
    "frequency": [
        "An ad appears every ten seconds. I cannot use the app like this.",
        "Sale un anuncio cada diez segundos. Así no puedo usar la app.",
        "Aparece um anúncio a cada dez segundos. Assim não consigo usar o app.",
        "Iklan muncul setiap sepuluh detik. Saya tidak bisa memakai aplikasi seperti ini.",
        "हर दस सेकंड में विज्ञापन आता है। ऐसे ऐप इस्तेमाल नहीं कर सकता।",
        "يظهر إعلان كل عشر ثوانٍ. لا أستطيع استخدام التطبيق هكذا.",
    ],
    "conditional_refund": [
        "The app keeps crashing. Fix it first; refund me only if it cannot be fixed.",
        "La app se cierra sola. Arréglenla primero; quiero un reembolso solo si no se puede arreglar.",
        "O app fecha sozinho. Corrijam primeiro; só quero reembolso se não houver conserto.",
        "Aplikasi terus menutup sendiri. Perbaiki dulu; kembalikan uang hanya jika tidak bisa diperbaiki.",
        "ऐप बार-बार बंद हो रहा है। पहले ठीक करें; ठीक न हो सके तभी पैसे लौटाएँ।",
        "يتعطل التطبيق باستمرار. أصلحوه أولاً، وأعيدوا المال فقط إن تعذر إصلاحه.",
    ],
    "correction": [
        "Correction: it is iOS, not Android. The app still freezes.",
        "Corrección: es iOS, no Android. La app sigue bloqueándose.",
        "Correção: é iOS, não Android. O app continua travando.",
        "Koreksi: ini iOS, bukan Android. Aplikasi masih macet.",
        "सुधार: यह iOS है, Android नहीं। ऐप अब भी अटक रहा है।",
        "تصحيح: النظام iOS وليس Android. التطبيق لا يزال يتجمد.",
    ],
    "resolved_billing": [
        "The duplicate charge is resolved now, but Premium still does not work.",
        "El cobro duplicado ya está resuelto, pero Premium sigue sin funcionar.",
        "A cobrança duplicada foi resolvida, mas o Premium ainda não funciona.",
        "Tagihan ganda sudah selesai, tetapi Premium masih tidak berfungsi.",
        "दोहरी कटौती का मामला सुलझ गया, लेकिन Premium अभी भी नहीं चलता।",
        "تم حل مشكلة الخصم المكرر، لكن Premium لا يعمل بعد.",
    ],
    "promise": [
        "The ad promised offline use, but the app requires internet for everything.",
        "El anuncio prometía uso sin conexión, pero la app necesita internet para todo.",
        "O anúncio prometia uso offline, mas o app exige internet para tudo.",
        "Iklan menjanjikan penggunaan offline, tetapi aplikasi membutuhkan internet untuk semuanya.",
        "विज्ञापन ने ऑफलाइन उपयोग का वादा किया था, लेकिन ऐप में हर काम के लिए इंटरनेट चाहिए।",
        "وعد الإعلان بالاستخدام دون اتصال، لكن التطبيق يحتاج الإنترنت لكل شيء.",
    ],
    "human": [
        "I cannot sign in. Please connect me to a human agent.",
        "No puedo iniciar sesión. Por favor, pásenme con una persona.",
        "Não consigo entrar. Por favor, quero falar com uma pessoa.",
        "Saya tidak bisa masuk. Tolong hubungkan saya dengan petugas manusia.",
        "मैं लॉगिन नहीं कर पा रहा हूँ। कृपया किसी मानव सहायता प्रतिनिधि से बात कराएँ।",
        "لا أستطيع تسجيل الدخول. أرجو تحويلي إلى موظف دعم بشري.",
    ],
    "reward": [
        "I watched the rewarded video, but I did not receive the coins.",
        "Vi el vídeo con recompensa, pero no recibí las monedas.",
        "Assisti ao vídeo de recompensa, mas não recebi as moedas.",
        "Saya menonton video berhadiah, tetapi koinnya tidak masuk.",
        "मैंने इनाम वाला वीडियो देखा, लेकिन सिक्के नहीं मिले।",
        "شاهدت فيديو المكافأة، لكنني لم أحصل على العملات.",
    ],
    "paid_ads": [
        "I paid to remove ads, but full-screen ads still appear.",
        "Pagué para quitar los anuncios, pero siguen apareciendo a pantalla completa.",
        "Paguei para remover anúncios, mas ainda aparecem anúncios em tela cheia.",
        "Saya membayar untuk menghapus iklan, tetapi iklan layar penuh tetap muncul.",
        "मैंने विज्ञापन हटाने के लिए भुगतान किया, लेकिन पूरे स्क्रीन के विज्ञापन अब भी आते हैं।",
        "دفعت لإزالة الإعلانات، لكنها لا تزال تظهر بملء الشاشة.",
    ],
    "negated_refund": [
        "I do not want a refund. Please restore my Premium access.",
        "No quiero un reembolso. Por favor, restauren mi acceso a Premium.",
        "Não quero reembolso. Por favor, restaurem meu acesso ao Premium.",
        "Saya tidak ingin pengembalian dana. Tolong pulihkan akses Premium saya.",
        "मुझे पैसे वापस नहीं चाहिए। कृपया मेरा Premium एक्सेस बहाल करें।",
        "لا أريد استرداد المال. أرجو استعادة وصولي إلى Premium.",
    ],
}


def evidence(message):
    return [{"message_id": message["id"], "quote": message["content"]}]


def issue(message, intent, aspect, status="unresolved", **slots):
    return {"intent": intent, "aspect": aspect, "sentiment": "negative", "status": status,
            "slots": {key: slots.get(key) for key in (
                "payment_channel", "payment_order_id", "amount", "currency", "app_version",
                "device_model", "os", "time_expression", "product")},
            "evidence_spans": evidence(message), "ad_format": None, "ad_placement": None,
            "advertised_claim": None, "reported_experience": None}


def build_records():
    records = []
    for group, translations in TEXTS.items():
        for language_index, (lang, text) in enumerate(zip(LANGUAGES, translations)):
            rid = f"{group}-{lang}"
            msg = {"id": "u1", "role": "user", "content": text}
            messages = [msg]
            target = {"schema_version": "1.0", "language": lang, "primary_intent": "unknown",
                      "issues": [], "request_conditions": [], "corrections": [],
                      "explicit_human_request": False, "human_request_evidence": [],
                      "urgency_signals": [], "missing_information": []}
            if group == "premium":
                target["issues"] = [issue(msg, "premium_not_activated", "premium", payment_channel="Stripe")]
                target["missing_information"] = ["payment_order_id"]
            elif group == "frequency":
                target["issues"] = [issue(msg, "ad_complaint", "ad_frequency")]
            elif group == "conditional_refund":
                target["issues"] = [issue(msg, "crash_performance", "crash"),
                                    issue(msg, "refund_request", "refund", "conditional")]
                target["request_conditions"] = [{"intent": "refund_request", "condition": text,
                                                 "evidence_spans": evidence(msg)}]
            elif group == "correction":
                messages = [{"id": "u0", "role": "user", "content": "Android"}, msg]
                target["issues"] = [issue(msg, "crash_performance", "performance", os="iOS")]
                target["corrections"] = [{"field": "os", "old_value": "Android", "new_value": "iOS",
                                          "evidence_spans": evidence(msg)}]
            elif group == "resolved_billing":
                messages = [{"id": "u0", "role": "user", "content": INITIAL_BILLING[language_index]},
                            {"id": "a0", "role": "assistant", "content": ASSISTANT_BILLING[language_index]}, msg]
                target["issues"] = [issue(msg, "premium_not_activated", "premium"),
                                    issue(msg, "subscription_billing", "billing", "resolved")]
            elif group == "promise":
                target["issues"] = [issue(msg, "ad_complaint", "ad_promise_mismatch")]
                target["issues"][0]["advertised_claim"] = CLAIMS[language_index]
                target["issues"][0]["reported_experience"] = EXPERIENCES[language_index]
            elif group == "human":
                target["issues"] = [issue(msg, "login_issue", "login")]
                target["explicit_human_request"] = True
                target["human_request_evidence"] = evidence(msg)
                target["urgency_signals"] = [{"kind": "blocked_access", "evidence_spans": evidence(msg)}]
            elif group == "reward":
                target["issues"] = [issue(msg, "ad_complaint", "ad_reward_missing")]
                target["issues"][0]["ad_format"] = "rewarded_video"
            elif group == "paid_ads":
                target["issues"] = [issue(msg, "ad_complaint", "paid_still_ads")]
                target["issues"][0]["ad_format"] = "interstitial"
            elif group == "negated_refund":
                target["issues"] = [issue(msg, "premium_not_activated", "premium"),
                                    issue(msg, "refund_request", "refund", "negated")]
            target["primary_intent"] = target["issues"][0]["intent"]
            record = DatasetRecord.model_validate({
                "id": rid, "scenario_group_id": group, "source": "hand-authored-synthetic-seed-v1",
                "synthetic": True, "review_status": "pending",
                "input": {"request_id": rid, "messages": messages}, "target": target,
            })
            validate_evidence(record.target, record.input)
            records.append(record)
    # Hard cases are curated separately; no random label/translation generation.
    extras = [
        ("mixed-hi-en", "mixed", "Premium के पैसे दे दिए but it is still locked. Please fix access.", "premium_not_activated", "premium"),
        ("mixed-es-en", "mixed", "Ya pagué Premium but the ads keep coming every minute.", "ad_complaint", "ad_frequency"),
        ("mixed-ar-en", "mixed", "دفعت الاشتراك but Premium is still locked.", "premium_not_activated", "premium"),
        ("sarcasm-en", "en", "Wonderful ads: another one every ten seconds. Such a relaxing way to never use the app.", "ad_complaint", "ad_frequency"),
        ("privacy-en", "en", "Why does the app need access to my contacts? I do not consent.", "privacy_permissions", "privacy"),
        ("unknown-en", "en", "That thing happened again. Can you help?", "unknown", "other"),
    ]
    for rid, lang, text, intent, aspect in extras:
        msg = {"id": "u1", "role": "user", "content": text}
        target = records[0].target.model_dump()
        target.update(language=lang, primary_intent=intent, issues=[issue(msg, intent, aspect)], missing_information=[])
        if intent == "unknown":
            target["issues"][0].update(sentiment="unknown", status="unknown")
        record = DatasetRecord.model_validate({"id": rid, "scenario_group_id": rid,
            "source": "hand-authored-synthetic-seed-v1", "synthetic": True, "review_status": "pending",
            "input": {"request_id": rid, "messages": [msg]}, "target": target})
        validate_evidence(record.target, record.input)
        records.append(record)
    return records


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    records = build_records()
    write_jsonl(root / "data/seeds/feedback.jsonl", records)
    print(f"Wrote {len(records)} synthetic, unreviewed seed records.")
