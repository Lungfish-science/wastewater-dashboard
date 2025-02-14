#!/usr/bin/env python3

from typing import Any

import yaml


def nav_button(target_id: str, btn_type: str, text: str) -> str:
    """
    Creates a navigation button (prev or next) for the carousel.
    """
    return (
        f'<button class="carousel-control-{btn_type}" type="button" '
        f'data-bs-target="#{target_id}" data-bs-slide="{btn_type}">'
        f'  <span class="carousel-control-{btn_type}-icon" aria-hidden="true"></span>'
        f'  <span class="visually-hidden">{text}</span>'
        f"</button>"
    )


def carousel_item(
    caption: str,
    image: str,
    link: str,
    index: int,
    interval: int,
) -> tuple[str, str]:
    """
    Creates the HTML for a single carousel item and its corresponding indicator button.
    """
    # Determine active classes/attributes for the first item.
    active_class = " active" if index == 0 else ""
    active_attr = ' class="active" aria-current="true"' if index == 0 else ""

    # Indicator button for this slide.
    button_html = (
        f'<button type="button" data-bs-target="#gallery-carousel" '
        f'data-bs-slide-to="{index}" aria-label="Slide {index + 1}"{active_attr}></button>'
    )

    # Carousel item (slide) content.
    item_html = (
        f'<div class="carousel-item{active_class}" data-bs-interval="{interval}">'
        f'  <a href="{link}"><img src="{image}" class="d-block mx-auto border" alt="{caption}"></a>'
        f'  <div class="carousel-caption d-none d-md-block">'
        f'      <p class="fw-light">{caption}</p>'
        f"  </div>"
        f"</div>"
    )

    return button_html, item_html


def carousel(carousel_id: str, duration: int, items: list[dict[str, Any]]) -> str:
    """
    Builds the full carousel HTML component.

    :param carousel_id: The id for the carousel.
    :param duration: The interval duration (in ms) for each slide.
    :param items: A list of dictionaries, each with keys 'caption', 'image', and 'link'.
    :return: A string of HTML for the carousel.
    """
    buttons = []
    items_html = []

    for index, item in enumerate(items):
        caption = item.get("caption", "")
        image = item.get("image", "")
        link = item.get("link", "#")
        btn, itm = carousel_item(caption, image, link, index, duration)
        buttons.append(btn)
        items_html.append(itm)

    # Group indicator buttons.
    indicators_html = '<div class="carousel-indicators">\n' + "\n".join(buttons) + "\n</div>"

    # Group carousel items.
    inner_html = '<div class="carousel-inner">\n' + "\n".join(items_html) + "\n</div>"

    # Navigation buttons.
    nav_prev = nav_button(carousel_id, "prev", "Previous")
    nav_next = nav_button(carousel_id, "next", "Next")

    # Full carousel container.
    return (
        f'<div id="{carousel_id}" class="carousel carousel-dark slide" data-bs-ride="carousel">\n'
        f"{indicators_html}\n"
        f"{inner_html}\n"
        f"{nav_prev}\n"
        f"{nav_next}\n"
        f"</div>"
    )


if __name__ == "__main__":
    # Load the carousel items from a YAML file.
    with open("carousel.yml") as f:
        items = yaml.safe_load(f)

    # Build the carousel HTML with an id and duration (e.g., 5000 ms).
    html_output = carousel("gallery-carousel", 5000, items)
