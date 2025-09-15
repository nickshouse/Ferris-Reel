use std::{collections::HashMap, path::PathBuf};

use crate::gui::{FileMeta, ImgEntry, SortKey};

#[inline]
pub fn sort_images(images: &mut [ImgEntry], key: SortKey, asc: bool) {
    images.sort_by(|a, b| {
        let ord = match key {
            SortKey::Name    => a.name.cmp(&b.name),
            SortKey::Created => a.created.cmp(&b.created),
            SortKey::Size    => a.bytes.cmp(&b.bytes),
            SortKey::Height  => a.h.cmp(&b.h).then(a.name.cmp(&b.name)),
            SortKey::Width   => a.w.cmp(&b.w).then(a.name.cmp(&b.name)),
            SortKey::Type    => a.ext.cmp(&b.ext).then(a.name.cmp(&b.name)),
        };
        if asc { ord } else { ord.reverse() }
    });
}

#[inline]
pub fn sort_files_lightweight(
    files: &mut [FileMeta],
    key: SortKey,
    asc: bool,
    index_of: &mut HashMap<PathBuf, usize>,
) {
    use SortKey::*;
    files.sort_by(|a, b| {
        let ord = match key {
            Name         => a.name.cmp(&b.name),
            Created      => a.created.cmp(&b.created),
            Size         => a.bytes.cmp(&b.bytes),
            Type         => a.ext.cmp(&b.ext).then(a.name.cmp(&b.name)),
            Height | Width => a.name.cmp(&b.name), // dims unknown until decode
        };
        if asc { ord } else { ord.reverse() }
    });

    index_of.clear();
    for (i, m) in files.iter().enumerate() {
        index_of.insert(m.path.clone(), i);
    }
}
