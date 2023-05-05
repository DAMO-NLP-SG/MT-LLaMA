def construct_input(lang, task, text1, text2):
    if lang == "ar":
        if task == 'cls':
            input_str = "حدد فئة النص من الاختيارات. الاختيارات: %s. النص: %s. الفئة:" % (
            text1, text2)
        elif task == 'ext':
            input_str = "تحديد الامتدادات من السياق وفقًا للاستعلام. استفسار: %s. سياق: %s. يمتد:" % (
                text1, text2)
    elif lang == "bn":
        if task == 'cls':
            input_str = "পছন্দ থেকে পাঠ্যের বিভাগ নির্ধারণ করুন। পছন্দ: %s পাঠ্য: %s বিভাগ:" % (
            text1, text2)
        elif task == 'ext':
            input_str = "ক্যোয়ারী অনুযায়ী প্রসঙ্গ থেকে স্প্যান সনাক্ত করুন। প্রশ্ন: %s প্রসঙ্গ: %s স্প্যান:" % (
                text1, text2)
    elif lang == "de":
        if task == 'cls':
            input_str = "Bestimmen Sie die Kategorie des Textes aus Auswahlmöglichkeiten. Auswahlmöglichkeiten: %s. Text: %s. Kategorie:" % (
            text1, text2)
        elif task == 'ext':
            input_str = "Identifizieren Sie Spannen aus dem Kontext gemäß der Abfrage. Anfrage: %s. Kontext: %s. Spannweiten:" % (
                text1, text2)
    elif lang == "fi":
        if task == 'cls':
            input_str = "Määritä tekstin luokka vaihtoehdoista. Vaihtoehdot: %s. Teksti: %s. Kategoria:" % (
            text1, text2)
        elif task == 'ext':
            input_str = "Tunnista jänteet kontekstista kyselyn mukaan. Kysely: %s. Konteksti: %s. Kantavuus:" % (
                text1, text2)
    elif lang == "fr":
        if task == 'cls':
            input_str = "Déterminez la catégorie du texte parmi les choix. Les choix: %s. Texte: %s. Catégorie:" % (
            text1, text2)
        elif task == 'ext':
            input_str = "Identifiez les étendues du contexte en fonction de la requête. Mettre en doute: %s. Contexte: %s. Portées:" % (
                text1, text2)
    elif lang == "el":
        if task == 'cls':
            input_str = "Προσδιορίστε την κατηγορία του κειμένου από επιλογές. Επιλογές: %s. Κείμενο: %s. Κατηγορία:" % (
            text1, text2)
        elif task == 'ext':
            input_str = "Προσδιορίστε τις εκτάσεις από το περιβάλλον σύμφωνα με το ερώτημα. Ερώτημα: %s. Συμφραζόμενα: %s. Εκτείνεται:" % (
                text1, text2)
    elif lang == "en":
        if task == 'cls':
            input_str = "Determine the category of the text from choices. Choices: %s. Text: %s. Category:" % (
            text1, text2)
        elif task == 'ext':
            input_str = "Identify spans from the context according to the query. Query: %s. Context: %s. Spans:" % (
                text1, text2)
    elif lang == "es":
        if task == 'cls':
            input_str = "Determinar la categoría del texto a partir de las opciones. Opciones: %s. Texto: %s. Categoría:" % (
            text1, text2)
        elif task == 'ext':
            input_str = "Identifique tramos del contexto de acuerdo con la consulta. Consulta: %s. Contexto: %s. Se extiende:" % (
                text1, text2)
    elif lang == "hi":
        if task == 'cls':
            input_str = "विकल्पों में से पाठ की श्रेणी निर्धारित करें। विकल्प: %s। मूलपाठ: %s। वर्ग:" % (
            text1, text2)
        elif task == 'ext':
            input_str = "क्वेरी के अनुसार संदर्भ से स्पैन की पहचान करें। जिज्ञासा: %s। प्रसंग: %s। स्पैन:" % (
                text1, text2)
    elif lang == "id":
        if task == 'cls':
            input_str = "Tentukan kategori teks dari pilihan. Pilihan: %s. Teks: %s. Kategori:" % (
            text1, text2)
        elif task == 'ext':
            input_str = "Identifikasi rentang dari konteks sesuai dengan kueri. Kueri: %s. Konteks: %s. Rentang:" % (
                text1, text2)
    elif lang == "it":
        if task == 'cls':
            input_str = "Determinare la categoria del testo dalle scelte. Scelte: %s. Testo: %s. Categoria:" % (
            text1, text2)
        elif task == 'ext':
            input_str = "Identifica gli intervalli dal contesto in base alla query. Domanda: %s. Contesto: %s. Campate:" % (
                text1, text2)
    elif lang == "ja":
        if task == 'cls':
            input_str = "選択肢からテキストのカテゴリを決定します。 選択肢：%s。 文章：%s。 カテゴリー：" % (
            text1, text2)
        elif task == 'ext':
            input_str = "クエリに従ってコンテキストからスパンを識別します。 クエリ：%s。 コンテクスト：%s。 スパン:" % (
                text1, text2)
    elif lang == "ko":
        if task == 'cls':
            input_str = "선택에서 텍스트의 범주를 결정합니다. 선택: %s. 텍스트: %s. 범주:" % (
            text1, text2)
        elif task == 'ext':
            input_str = "쿼리에 따라 컨텍스트에서 범위를 식별합니다. 쿼리: %s. 문맥: %s. 스팬:" % (
                text1, text2)
    elif lang == "nl":
        if task == 'cls':
            input_str = "Bepaal de categorie van de tekst uit keuzes. Keuzes: %s. Tekst: %s. Categorie:" % (
            text1, text2)
        elif task == 'ext':
            input_str = "Identificeer overspanningen uit de context volgens de query. Vraag: %s. Context: %s. Overspanningen:" % (
                text1, text2)
    elif lang == "pl":
        if task == 'cls':
            input_str = "Określ kategorię tekstu z wyborów. Wybory: %s. Tekst: %s. Kategoria:" % (
            text1, text2)
        elif task == 'ext':
            input_str = "Zidentyfikuj rozpiętości z kontekstu zgodnie z zapytaniem. Zapytanie: %s. Kontekst: %s. Rozpiętości:" % (
                text1, text2)
    elif lang == "pt":
        if task == 'cls':
            input_str = "Determine a categoria do texto a partir das opções. Escolhas: %s. Texto: %s. Categoria:" % (
            text1, text2)
        elif task == 'ext':
            input_str = "Identifique spans a partir do contexto de acordo com a consulta. Consulta: %s. Contexto: %s. Períodos:" % (
                text1, text2)
    elif lang == "ru":
        if task == 'cls':
            input_str = "Определите категорию текста из вариантов. Выбор: %s. Текст: %s. Категория:" % (
            text1, text2)
        elif task == 'ext':
            input_str = "Идентифицируйте промежутки из контекста в соответствии с запросом. Запрос: %s. Контекст: %s. Пролеты:" % (
                text1, text2)
    elif lang == "sv":
        if task == 'cls':
            input_str = "Bestäm kategorin för texten från val. Alternativ: %s. Text: %s. Kategori:" % (
            text1, text2)
        elif task == 'ext':
            input_str = "Identifiera spann från sammanhanget enligt frågan. Fråga: %s. Sammanhang: %s. Spännvidder:" % (
                text1, text2)
    elif lang == "sw":
        if task == 'cls':
            input_str = "Amua aina ya maandishi kutoka kwa chaguo. Chaguo: %s. Maandishi: %s. Kategoria:" % (
            text1, text2)
        elif task == 'ext':
            input_str = "Tambua vipindi kutoka kwa muktadha kulingana na hoja. Swali: %s. Muktadha: %s. Vipindi:" % (
                text1, text2)
    elif lang == "te":
        if task == 'cls':
            input_str = "ఎంపికల నుండి టెక్స్ట్ యొక్క వర్గాన్ని నిర్ణయించండి. ఎంపికలు: %s. వచనం: %s. వర్గం:" % (
            text1, text2)
        elif task == 'ext':
            input_str = "ప్రశ్నకు అనుగుణంగా సందర్భం నుండి పరిధులను గుర్తించండి. ప్రశ్న: %s. సందర్భం: %s. పరిధులు:" % (
                text1, text2)
    elif lang == "th":
        if task == 'cls':
            input_str = "กำหนดหมวดหมู่ของข้อความจากตัวเลือก ตัวเลือก: %s ข้อความ: %s. หมวดหมู่:" % (
            text1, text2)
        elif task == 'ext':
            input_str = "ระบุช่วงจากบริบทตามแบบสอบถาม ข้อความค้นหา: %s บริบท: %s. ช่วง:" % (
                text1, text2)
    elif lang == "tr":
        if task == 'cls':
            input_str = "Seçeneklerden metnin kategorisini belirleyin. Seçenekler: %s. Metin: %s. Kategori:" % (
            text1, text2)
        elif task == 'ext':
            input_str = "Sorguya göre bağlamdan açıklıkları tanımlayın. Sorgu: %s. Bağlam: %s. açıklıklar:" % (
                text1, text2)
    elif lang == "vi":
        if task == 'cls':
            input_str = "Xác định thể loại của văn bản từ các lựa chọn. Lựa chọn: %s. Chữ: %s. Loại:" % (
            text1, text2)
        elif task == 'ext':
            input_str = "Xác định các khoảng từ ngữ cảnh theo truy vấn. Truy vấn: %s. Bối cảnh: %s. nhịp:" % (
                text1, text2)
    elif lang == "zh":
        if task == 'cls':
            input_str = "根据选项确定文本的类别。选项：%s。文本：%s。类别：" % (
            text1, text2)
        elif task == 'ext':
            input_str = "根据查询从上下文中识别跨度。查询：%s。上下文：%s。跨度：" % (
                text1, text2)
    else:
        if task == 'cls':
            input_str = "Determine the category of the text from choices. Choices: %s. Text: %s. Category:" % (
                text1, text2)
        elif task == 'ext':
            input_str = "Identify spans from the context according to the query. Query: %s. Context: %s. Spans:" % (
                text1, text2)
    return input_str

class S2SFeatures:
    '''
    MRC features
    '''
    def __init__(
        self,
        query,
        context,
        answer,
        target_entity,
        len_query,
        len_context,
        lang=None,
    ):
        self.query = query
        self.context = context
        self.answer = answer
        self.target_entity = target_entity
        self.len_query = len_query
        self.len_context = len_context
        self.lang = lang