import os
import re
import time

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.core.os_manager import ChromeType

# ===== НАСТРОЙКИ =====
BASE_URL = "https://www.championat.ru"
LIST_TEMPLATE = "https://www.championat.ru/news/basketball/_nba/{}.html"
OUTPUT_DIR = "dataset"
PAGES = [1, 2, 3]

# Путь к Brave (macOS)
BRAVE_PATH = "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser"

# === ВАЖНО: Укажите вашу версию Brave (Chromium) ===
# Текущая версия из ошибки: 147.0.7727.102
# Основная версия Chromium для совместимости драйвера: 147
BRAVE_VERSION = "147"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===== ЗАПУСК БРАУЗЕРА =====
options = Options()
options.binary_location = BRAVE_PATH
# options.add_argument("--headless")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36")

# Автоматическая загрузка драйвера точно под версию Brave
service = Service(
    ChromeDriverManager(
        chrome_type=ChromeType.BRAVE,
        driver_version=BRAVE_VERSION  # Явно указываем версию
    ).install()
)

driver = webdriver.Chrome(service=service, options=options)

all_links = []

try:
    # 1. Собираем все ссылки с трёх страниц списка
    for page in PAGES:
        driver.get(LIST_TEMPLATE.format(page))
        wait = WebDriverWait(driver, 15)
        wait.until(EC.presence_of_element_located(
            (By.CSS_SELECTOR, ".page-content .news-item__content")
        ))
        time.sleep(1)

        blocks = driver.find_elements(By.CSS_SELECTOR, ".page-content .news-item__content")
        for block in blocks:
            try:
                link = block.find_element(By.CSS_SELECTOR, "a.news-item__title").get_attribute("href")
                if link:
                    if link.startswith("/"):
                        link = BASE_URL + link
                    all_links.append(link)
            except Exception as e:
                print(f"Пропущен блок (нет ссылки): {e}")

    print(f"Найдено новостей: {len(all_links)}")

    # 2. Переходим по каждой ссылке, извлекаем заголовок и текст
    for idx, url in enumerate(all_links, start=1):
        print(f"[{idx}/{len(all_links)}] Открываю: {url}")
        driver.get(url)

        try:
            wait.until(EC.presence_of_element_located(
                (By.CSS_SELECTOR, ".article-head__title")
            ))
            time.sleep(0.5)

            # Заголовок
            title = driver.find_element(By.CSS_SELECTOR, ".article-head__title").text.strip()

            # Основной текст (все абзацы внутри .article-content)
            paragraphs = driver.find_elements(By.CSS_SELECTOR, ".article-content p")
            content = "\n\n".join(p.text.strip() for p in paragraphs if p.text.strip())

            # Формируем Markdown
            md_content = f"# {title}\n\n{content}\n"

            # Безопасное имя файла
            safe_title = re.sub(r'[\\/*?:"<>|]', "", title)
            safe_title = safe_title[:80].strip()
            filename = f"{idx:02d}_{safe_title}.md"
            filepath = os.path.join(OUTPUT_DIR, filename)

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(md_content)

            print(f"✔ Сохранён: {filepath}")

        except Exception as e:
            print(f"Ошибка при обработке {url}: {e}")

finally:
    driver.quit()
    print("Готово. Браузер закрыт.")