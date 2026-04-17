use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use itertools::Itertools;
use serde::{Deserialize, Serialize};

use crate::paths::SanitizedPath;

/// A list of absolute paths, in a specific order.
///
/// The paths are stored in lexicographic order, so that they can be compared to
/// other path lists without regard to the order of the paths.
///
/// The paths can be retrieved in the original order using `ordered_paths()`.
#[derive(Default, PartialEq, Eq, Hash, Debug, Clone, Serialize, Deserialize)]
pub struct PathList {
    /// The paths, in lexicographic order.
    paths: Arc<[PathBuf]>,
    /// The order in which the paths were provided.
    ///
    /// See `ordered_paths()` for a way to get the paths in the original order.
    order: Arc<[usize]>,
}

#[derive(Debug)]
pub struct SerializedPathList {
    pub paths: String,
    pub order: String,
}

impl PathList {
    pub fn new<P: AsRef<Path>>(paths: &[P]) -> Self {
        let mut indexed_paths: Vec<(usize, PathBuf)> = paths
            .iter()
            .enumerate()
            .map(|(ix, path)| (ix, SanitizedPath::new(path).into()))
            .collect();
        indexed_paths.sort_by(|(_, a), (_, b)| a.cmp(b));
        let order = indexed_paths.iter().map(|e| e.0).collect::<Vec<_>>().into();
        let paths = indexed_paths
            .into_iter()
            .map(|e| e.1)
            .collect::<Vec<_>>()
            .into();
        Self { order, paths }
    }

    pub fn is_empty(&self) -> bool {
        self.paths.is_empty()
    }

    pub fn paths(&self) -> &[PathBuf] {
        self.paths.as_ref()
    }

    pub fn order(&self) -> &[usize] {
        self.order.as_ref()
    }

    pub fn ordered_paths(&self) -> impl Iterator<Item = &PathBuf> {
        self.order
            .iter()
            .zip(self.paths.iter())
            .sorted_by_key(|(i, _)| **i)
            .map(|(_, path)| path)
    }

    pub fn is_lexicographically_ordered(&self) -> bool {
        self.order.iter().enumerate().all(|(i, &j)| i == j)
    }

    pub fn deserialize(serialized: &SerializedPathList) -> Self {
        let mut paths: Vec<PathBuf> = if serialized.paths.is_empty() {
            Vec::new()
        } else {
            serialized.paths.split('\n').map(PathBuf::from).collect()
        };

        let mut order: Vec<usize> = serialized
            .order
            .split(',')
            .filter_map(|s| s.parse().ok())
            .collect();

        if !paths.is_sorted() || order.len() != paths.len() {
            order = (0..paths.len()).collect();
            paths.sort();
        }

        Self {
            paths: paths.into(),
            order: order.into(),
        }
    }

    pub fn serialize(&self) -> SerializedPathList {
        use std::fmt::Write as _;

        let mut paths = String::new();
        for path in self.paths.iter() {
            if !paths.is_empty() {
                paths.push('\n');
            }
            paths.push_str(&path.to_string_lossy());
        }

        let mut order = String::new();
        for ix in self.order.iter() {
            if !order.is_empty() {
                order.push(',');
            }
            write!(&mut order, "{}", *ix).unwrap();
        }
        SerializedPathList { paths, order }
    }
}
