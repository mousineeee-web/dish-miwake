"""Versioned dish candidates; adding a phrase does not train model weights."""

CATALOG_VERSION = 'open60-v3'
# key, Japanese display label, English visual description, Commons search query
ORIGINAL_DISHES = (
    ('sushi', '寿司', 'sushi', 'sushi plate'),
    ('ramen', 'ラーメン', 'ramen noodle soup', 'ramen bowl'),
    ('curry_rice', 'カレーライス', 'Japanese curry with rice', 'Japanese curry rice'),
    ('fried_rice', 'チャーハン', 'fried rice', 'fried rice'),
    ('pizza', 'ピザ', 'pizza', 'pizza margherita'),
    ('hamburger', 'ハンバーガー', 'a hamburger in a bun', 'hamburger food'),
    ('spaghetti', 'スパゲッティ', 'spaghetti pasta', 'spaghetti plate'),
    ('udon', 'うどん', 'Japanese udon thick white noodles', 'udon noodles'),
    ('soba', 'そば', 'Japanese soba buckwheat noodles', 'soba noodles'),
    ('yakisoba', '焼きそば', 'Japanese yakisoba stir-fried noodles', 'yakisoba'),
    ('gyoza', '餃子', 'Japanese pan-fried gyoza dumplings', 'gyoza dumplings'),
    ('takoyaki', 'たこ焼き', 'Japanese takoyaki octopus balls', 'takoyaki'),
    ('okonomiyaki', 'お好み焼き', 'Japanese okonomiyaki savory pancake', 'okonomiyaki'),
    ('tonkatsu', 'とんかつ', 'Japanese breaded pork cutlet tonkatsu', 'tonkatsu'),
    ('tempura', '天ぷら', 'Japanese tempura fried seafood and vegetables', 'tempura'),
    ('omurice', 'オムライス', 'Japanese omurice omelette over rice', 'omurice'),
    ('onigiri', 'おにぎり', 'Japanese onigiri rice balls with seaweed', 'onigiri'),
    ('steak', 'ステーキ', 'grilled beef steak', 'beef steak plate'),
    ('salad', 'サラダ', 'a green vegetable salad', 'green salad'),
    ('sandwich', 'サンドイッチ', 'a sandwich made with sliced bread', 'sandwich food'),
    ('french_fries', 'フライドポテト', 'French fries', 'French fries'),
    ('karaage', '唐揚げ', 'Japanese karaage fried chicken pieces', 'karaage'),
    ('gyudon', '牛丼', 'Japanese gyudon beef bowl over rice', 'gyudon'),
    ('oyakodon', '親子丼', 'Japanese oyakodon chicken and egg over rice', 'oyakodon'),
    ('hot_pot', '鍋料理', 'an Asian hot pot stew', 'hot pot food'),
    ('mapo_tofu', '麻婆豆腐', 'Chinese mapo tofu', 'mapo tofu'),
    ('peking_duck', '北京ダック', 'Chinese Peking roast duck', 'Peking duck dish'),
    ('dim_sum', '点心（焼売・蒸し餃子）', 'Chinese steamed dim sum dumplings', 'dim sum shumai'),
    ('cake', 'ケーキ', 'a slice of cake', 'slice cake'),
    ('croissant', 'クロワッサン', 'a French croissant pastry', 'croissant pastry'),
    ('lanzhou_beef_noodles', '蘭州牛肉麺', 'Chinese hand-pulled beef noodle soup', 'Lanzhou lamian'),
)

FRUITS = (
    ('apple', 'りんご', 'an apple fruit', 'apple fruit'),
    ('banana', 'バナナ', 'a banana fruit', 'banana fruit'),
    ('orange', 'オレンジ', 'an orange citrus fruit', 'orange fruit'),
    ('grapes', 'ぶどう', 'a bunch of grapes', 'grapes fruit'),
    ('strawberry', 'いちご', 'strawberries', 'strawberries'),
    ('watermelon', 'スイカ', 'a watermelon or slices of watermelon', 'watermelon slices'),
    ('pineapple', 'パイナップル', 'a pineapple fruit', 'pineapple fruit'),
    ('mango', 'マンゴー', 'a mango fruit or sliced mango', 'mango fruit'),
    ('kiwi', 'キウイ', 'a kiwi fruit cut in half showing green flesh and black seeds', 'kiwi fruit'),
    ('peach', '桃', 'a peach fruit', 'peach fruit'),
    ('pear', '梨', 'a pear fruit', 'pear fruit'),
    ('lemon', 'レモン', 'a lemon citrus fruit', 'lemon fruit'),
)
DRINKS = (
    ('water', '水（見た目の候補）', 'a glass or a bottle of clear drinking water', 'glass water'),
    ('tea', 'お茶（緑茶・紅茶など）', 'tea in a cup or a glass', 'cup tea'),
    ('coffee', 'コーヒー', 'coffee in a cup or a glass', 'cup coffee'),
    ('milk', '牛乳（見た目の候補）', 'a glass of white milk', 'glass milk'),
    ('orange_juice', 'オレンジジュース', 'a glass of orange juice', 'glass orange juice'),
    ('cola', 'コーラ', 'a glass of dark cola soda with bubbles', 'glass cola'),
    ('bubble_tea', 'タピオカミルクティー', 'a cup of bubble milk tea with tapioca pearls', 'bubble milk tea'),
    ('thai_iced_tea', 'タイミルクティー', 'Thai iced tea with orange colored milk tea and ice', 'Thai iced tea'),
)
THAI_DISHES = (
    ('pad_thai', 'パッタイ', 'Thai pad thai stir fried rice noodles with bean sprouts and peanuts', 'pad thai'),
    ('thai_green_curry', 'タイグリーンカレー', 'Thai green curry with coconut milk and vegetables', 'Thai green curry'),
    ('thai_red_curry', 'タイレッドカレー', 'Thai red curry with coconut milk and vegetables', 'Thai red curry'),
    ('tom_yum', 'トムヤムクン', 'Thai tom yum spicy sour shrimp soup', 'tom yum soup'),
    ('tom_kha_gai', 'トムカーガイ', 'Thai tom kha gai white coconut chicken soup', 'tom kha gai'),
    ('gaprao', 'ガパオライス', 'Thai pad krapow stir fried basil minced meat with rice and a fried egg', 'pad krapow rice'),
    ('khao_man_gai', 'カオマンガイ', 'Thai khao man gai sliced poached chicken with rice and cucumber', 'khao man gai'),
    ('som_tam', 'ソムタム', 'Thai som tam shredded green papaya salad', 'som tam papaya salad'),
    ('mango_sticky_rice', 'マンゴースティッキーライス', 'Thai mango sticky rice dessert with coconut milk', 'mango sticky rice'),
)

# Keep the original 31 candidates, their order and descriptions unchanged.
DISHES = ORIGINAL_DISHES + FRUITS + DRINKS + THAI_DISHES
CATALOG_GROUPS = (('これまでの料理', ORIGINAL_DISHES), ('フルーツ', FRUITS),
                  ('飲み物', DRINKS), ('タイ料理', THAI_DISHES))
DRINK_KEYS = frozenset(row[0] for row in DRINKS)

FOOD_PROMPTS = (
    'a photo of a prepared meal on a plate',
    'a photo of food in a bowl',
    'a photo of a dessert or a snack',
    # These are now in-scope food groups, not evidence for out-of-scope rejection.
    'a photo of fresh fruit',
    'a photo of a drink in a glass or a cup',
)
OTHER_FOOD_PROMPTS = (
    ('raw_vegetables', 'a photo of raw vegetables'),
    ('bread', 'a photo of a loaf of bread or a baguette'),
    ('pancakes', 'a photo of breakfast pancakes or waffles'),
    ('paella', 'a photo of Spanish paella'),
    ('pho', 'a photo of Vietnamese pho noodle soup'),
    ('macarons', 'a photo of French macarons'),
)
NON_FOOD_PROMPTS = (
    ('flowers', 'a photo of flowers in a garden or a field'),
    ('animal', 'a photo of a dog or a cat'),
    ('person', 'a photo of a person'),
    ('vehicle', 'a photo of a car or a bicycle'),
    ('landscape', 'a photo of a mountain, a forest or a beach'),
    ('building', 'a photo of a building or a street'),
    ('document', 'a photo of a printed document or a page of text'),
    ('screen', 'a screenshot of a computer or phone screen'),
    ('household', 'a photo of furniture or household objects'),
    ('electronics', 'a photo of an electronic device'),
    ('clothes', 'a photo of clothes or shoes'),
    ('drawing', 'a drawing, a painting or a cartoon'),
    ('empty_plate', 'a photo of an empty plate, bowl or cup'),
    ('pattern', 'an abstract pattern, noise or a blank image'),
    ('utensil', 'a photo of cutlery, a spoon, a fork or a kitchen utensil'),
)


def dish_prompts(language):
    if language not in ('English', '日本語'):
        raise ValueError(f'Unsupported prompt language: {language}')
    return tuple('a photo of ' + english if language == 'English' else japanese + 'の写真'
                 for _, japanese, english, _ in DISHES)


def all_prompts(language):
    return dish_prompts(language) + FOOD_PROMPTS + tuple(p for _, p in OTHER_FOOD_PROMPTS + NON_FOOD_PROMPTS)
