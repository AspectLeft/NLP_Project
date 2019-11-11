from data.semeval_2014_xml_loader import *


class TextOnlyLoader(BaseLoader):
    @staticmethod
    def handle_text(text):
        print(text)

    @staticmethod
    def handle_aspect_term(attrs):
        pass

    @staticmethod
    def handle_aspect_category(attrs):
        pass


if __name__ == '__main__':
    main(TextOnlyLoader)

