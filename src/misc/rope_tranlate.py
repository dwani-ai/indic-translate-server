import torch
import warnings
from IndicTransToolkit.processor import IndicProcessor
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

warnings.filterwarnings("ignore")
model_name = "prajdabre/rotary-indictrans2-en-indic-1B"
device = "cuda" if torch.cuda.is_available() else "cpu"

sentences = [
    """As I wandered through the bustling city streets on a warm, sunny afternoon, the world around me seemed to hum with life. The vibrant colors of flowers blooming in the meticulously maintained parks painted a vivid contrast against the steel and glass of the towering skyscrapers that loomed above, reflecting the bright blue sky in their countless windows. The sunlight, filtering through the canopy of leaves in the small, scattered patches of greenery that dotted the city, cast dappled shadows on the pavement below, where the rhythmic clatter of footsteps and the distant hum of traffic merged into a soothing, urban symphony. Street performers, positioned at nearly every corner, played their guitars, violins, and saxophones, their melodies blending into a medley of sounds that flowed like a river through the streets, carried by the laughter and chatter of people spilling out of bustling cafés and restaurants, their faces alight with the joy of an afternoon well spent. Children raced through the playgrounds, their laughter carried on the gentle breeze that occasionally swept through the streets, bringing with it the scent of freshly baked bread from a nearby bakery and the tang of street food sizzling on grills as vendors called out to passersby. In the midst of all this vibrant activity, I found myself swept up in the energy of the city, its pulse echoing in my ears as I walked along, watching the ever-changing tapestry of life unfold before me. The city was a living, breathing organism, constantly in motion, and I couldn’t help but marvel at how different this world felt from the one I had known for so long. It hadn’t been that long ago, only a few months, that I had lived in the countryside, surrounded by rolling hills and quiet fields that stretched as far as the eye could see. The mornings there had always been peaceful, almost meditative in their stillness, with only the soft rustle of leaves in the trees, the gentle whisper of the wind as it passed through the tall grasses, and the occasional chirping of birds breaking the silence. I would wake early, just as the first rays of sunlight crept over the horizon, casting a soft, golden light over the landscape. The air would be cool and crisp, filled with the earthy scent of dew-soaked soil and the faint aroma of wildflowers. Those mornings had their own kind of beauty, a beauty that came from the quiet simplicity of nature, from the feeling of being alone in the world, surrounded by nothing but the sounds of the earth waking up. I would sit on the porch of my small cottage, with a cup of steaming coffee in hand, and watch as the mist slowly lifted from the fields, revealing the distant outline of the forest, where the trees stood tall and silent, like sentinels guarding the secrets of the land. Sometimes, if I was lucky, I’d catch a glimpse of a deer or two grazing in the distance, their sleek bodies moving gracefully through the tall grass, unaware of my presence. But despite the peace and tranquility of the countryside, there had always been a part of me that longed for something more, something beyond the quiet, predictable rhythm of rural life. I had spent years surrounded by nature’s beauty, and while I had loved it, I had also begun to feel a sense of restlessness, a yearning for the energy and excitement that only a city could provide. So, when the opportunity came to move to the city, I had taken it without hesitation, packing up my life into a few boxes and leaving behind the familiar comforts of the countryside for the unknown adventures that awaited me in the bustling streets of urban life. Now, as I walked through those very streets, I realized just how much my life had changed in such a short time. The cit y was everything I had hoped it would be, and more. Every day brought something new, something unexpected. One day, I might stumble upon a hidden café tucked away in a narrow alley, its walls covered in ivy, serving the best espresso I had ever tasted. The next, I might find myself standing in the middle of a street festival, surrounded by food stalls offering dishes from every corner of the world, the air filled with the scent of spices and grilled meats, while performers danced and played music in the center of the crowd, their movements a celebration of culture and tradition. The people, too, were different. In the countryside, I had known everyone—every face was familiar, every story already told. But here, in the city, every person I passed was a mystery, a new story waiting to be discovered. There were the artists, sitting in cafés sketching scenes from their imagination; the businesspeople, hurrying to meetings with phones pressed to their ears, their faces a mask of determination; the students, gathered in groups, discussing everything from philosophy to the latest fashion trends; and the tourists, cameras slung around their necks, eyes wide with wonder as they took in the sights of the city. Each of them was on their own journey, their paths crossing with mine for just a brief moment before they continued on their way, leaving behind only the faintest trace of their presence. Yet, despite all the excitement and energy of the city, there were still moments when I found myself missing the quiet solitude of the countryside. There were times, late at night, when the city had finally quieted down, and I would lie awake in my small apartment, listening to the distant sounds of cars passing by on the streets below, that I would think back to those early mornings in the countryside, to the feeling of peace that came from being alone with nature. I would remember the way the first light of dawn had turned the sky a soft shade of pink, the way the wind had whispered through the trees, and the way the world had felt so still, as if it were holding its breath, waiting for the day to begin. But then the sun would rise, casting its golden light over the city once again, and I would step outside, greeted by the sights and sounds of life unfolding around me, and I would remember why I had chosen to leave the quiet behind. The city, with all its noise, its chaos, and its constant movement, had a magic of its own, a magic that drew me in and made me feel alive in a way that the countryside never could. It was a place where anything seemed possible, where every corner held the promise of something new, something unexpected. And as I continued to walk through the streets that day, I knew that, despite the occasional longing for the peace of the countryside, I had found a new home in the heart of the city, where the pulse of life beat strong and steady, carrying me along with it."""
]

ip = IndicProcessor(inference=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    trust_remote_code=True,
).to(device)

batch = ip.preprocess_batch(sentences, src_lang="eng_Latn", tgt_lang="hin_Deva")

batch = tokenizer(
    batch, padding="longest", truncation=True, max_length=2048, return_tensors="pt"
).to(device)

with torch.inference_mode():
    outputs = model.generate(
        **batch,
        num_beams=10,
        length_penalty=1.5,
        repetition_penalty=2.0,
        num_return_sequences=1,
        max_new_tokens=2048,
        early_stopping=True
    )

# no target_tokenizer scoping is required anymore
outputs = tokenizer.batch_decode(
    outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True
)

outputs = ip.postprocess_batch(outputs, lang="hin_Deva")
print(" | > Translations:", outputs[0])
