import os
from aiogram import Bot, Dispatcher, executor
from aiogram.types import Message, ContentType
from antispoof import antispoof
import dotenv


dotenv.load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")

if not BOT_TOKEN:
    raise EnvironmentError("BOT_TOKEN is not provided")

bot = Bot(token=BOT_TOKEN)

dp = Dispatcher(bot)

model = antispoof.AntiSpoof()

print("Model loaded")

HELLO_PHRASE = """
Привет, этот бот использует анти-спуффинговую модель. Он может отличить настоящее лицо, смотрящее в камеру, от подставленной вместо него фотографии

Отправьте ему фото, видео или кружочек, тогда бот проверит файл на предмет спуфинга. Рекомендуется отправлять короткие видео, чтобы уменьшить время ожидания.
*Точность работы на видео больше в связи возможностью анализа сразу нескольких кадров.*

Обозначения:
🟩 Зеленая рамка вокруг лица означает, что оно настоящее. 
🟥 Красная рамка вокруг лица означает, что перед нами поддельное лицо.
🟨 Желтая рамка означает, что модель анализирует лицо. 

Если на видео лицо постоянно обведено желтым или рамки нет, необходимо сделать фотографию в более хорошем качестве и с лучшим освещением. 

Приятного пользования!
"""

FACE_NOT_FOUND_PHRASE = "🔍Лицо не найдено. Попробуйте cделать фото с хорошим освещением"

PROCESSING_PHRASE = "⏳Видео обрабатывается, это может занять некоторое время"


@dp.message_handler(commands=["start", "help"])
async def handle_start(message: Message):
    await message.reply( HELLO_PHRASE, parse_mode="Markdown")


@dp.message_handler(content_types=ContentType.PHOTO)
async def handle_photo(message: Message):
    photo = message.photo[-1]

    photo_path = (await photo.download(destination_dir="tmp/")).name

    photo_path = model.get_processed_photo_path(photo_path)

    if photo_path is None:
        await message.reply( FACE_NOT_FOUND_PHRASE)
        return

    await message.reply_photo(photo=open(photo_path, "rb"))


@dp.message_handler(content_types=ContentType.VIDEO)
async def handle_video(message: Message):
    video = message.video

    video_path = (await video.download(destination_dir="tmp/")).name

    await message.reply(PROCESSING_PHRASE)

    video_path = model.get_processed_video_path(video_path)

    if video_path is None:
        await message.reply(FACE_NOT_FOUND_PHRASE)
        return

    await message.reply_video(video=open(video_path, "rb"))


@dp.message_handler(content_types=ContentType.VIDEO_NOTE)
async def handle_video_note(message: Message):
    video_note = message.video_note

    video_note_path = (await video_note.download(destination_dir="tmp/")).name

    await message.reply(PROCESSING_PHRASE)

    video_note_path = model.get_processed_video_path(video_note_path)

    if video_note_path is None:
        await message.reply(FACE_NOT_FOUND_PHRASE)
        return

    await message.reply_video_note(video_note=open(video_note_path, "rb"))


if __name__ == "__main__":
    executor.start_polling(dp)
