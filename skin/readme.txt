========================================
  LWMP skin system - How To Make Skins
========================================

Skins let you customize the look of the piano roll's
keyboard and note textures. Each skin is a collection
of PNG image files.


QUICK START
-----------
1. Copy the "default" folder and rename it (e.g. "myskin").
2. Edit the PNG images inside your new folder.
3. Launch LWMP and press on the Skins button
4. Choose the skin you made. Thats it!


REQUIRED TEXTURES
-----------------
Every skin must contain these 6 PNG files:

  keyBlack.png          - Black piano key
  keyBlackPressed.png   - Black piano key (pressed state)
  keyWhite.png          - White piano key
  keyWhitePressed.png   - White piano key (pressed state)
  note.png              - Note body texture
  noteEdge.png          - Note edge/cap texture

File names are case-sensitive. If any file is missing,
that part will fall back to solid colors or the default
skin.


IMAGE GUIDELINES
----------------
- Format: PNG with alpha transparency (RGBA).

- Key textures: width/height ratio determines how keys
  are scaled. Keep proportions consistent.

- Note textures: tiled horizontally along the note length.
  noteEdge.png is drawn at both ends of each note.

- There are no strict size requirements, but keep them
  reasonable for performance (e.g. 64-256px wide).


DISTRIBUTING SKINS
------------------
You have two options for sharing skins:

  Option 1 - Folder
  -----------------
  Just zip up your skin folder and share it. The user
  extracts it into the skin/ directory so it looks like:

      skin/
        myskin/
          keyBlack.png
          keyBlackPressed.png
          keyWhite.png
          keyWhitePressed.png
          note.png
          noteEdge.png

  Option 2 - Zip File
  -------------------
  Create a .zip containing the 6 PNG files (either at
  the root of the zip, or inside a subfolder). Place it
  in the skin/ directory:

      skin/
        myskin.zip

  LWMP will automatically extract it to a cache folder
  on first use. The zip file name (without .zip) becomes
  the skin name used in config.json.


SKIN FOLDER STRUCTURE
---------------------
    skin/
      default/           <- built-in skin (do not delete)
        keyBlack.png
        keyBlackPressed.png
        keyWhite.png
        keyWhitePressed.png
        note.png
        noteEdge.png
      myskin/            <- folder-based custom skin
        ...
      anotherskin.zip    <- zip-based custom skin
      .cache/            <- auto-generated (do not touch)
      readme.txt         <- this file

The .cache/ folder is created automatically when a zip
skin is used. Do not edit or delete it manually.


TIPS
----
- Use an image editor that supports transparency
  (GIMP, Photoshop, Aseprite, etc.)
- For a clean pixel-art look, use nearest-neighbor
  scaling. For smooth textures, use bilinear.
- You can preview changes by restarting LWMP.
- Keep a backup of your original textures before
  editing the default skin.

========================================
