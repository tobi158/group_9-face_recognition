from icrawler.builtin import BingImageCrawler
import os

def download_images_for_labels(label_list, output_dir=r"C:\Users\BHXH\Desktop\venv_demo\fileTestAll", limit=10):
    os.makedirs(output_dir, exist_ok=True)

    for label in label_list:
        print(f"📥 Đang tải ảnh cho: {label}")
        label_folder = os.path.join(output_dir, label.replace(" ", "_"))
        os.makedirs(label_folder, exist_ok=True)

        crawler = BingImageCrawler(storage={"root_dir": label_folder})
        crawler.crawl(keyword=label + " face", max_num=limit)

if __name__ == "__main__":
    labels = [
        "Vicente Fox",
        "Tony Blair",
        "Andre Agassi",
        "Ariel Sharon",
        "Arnold Schwarzenegger",
        "David Beckham",
        "George W Bush",
        "Serena Williams",
        "Vladimir Putin",
        "Tom Ridge",
        "Silvio Berlusconi",
        "Roh Moo-hyun",
        "Recep Tayyip Erdogan",
        "Nestor Kirchner",
        "Megawati Sukarnoputri",
        "Luiz Inacio Lula da Silva",
        "Lleyton Hewitt",
        "Laura Bush",
        "Kofi Annan",
        "Junichiro Koizumi",
        "John Negroponte",
        "John Ashcroft",
        "Jennifer Capriati",
        "Jean Chretien",
        "Jacques Chirac",
        "Hugo Chavez",
        "Hans Blix",
        "Guillermo Coria",
        "Gloria Macapagal Arroyo",
        "Gerhard Schroeder",
        "Donald Rumsfeld",
        "Colin Powell"
    ]

    images_per_label = 1
    download_images_for_labels(labels, limit=images_per_label)
