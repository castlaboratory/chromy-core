//! Ordering the frame before selection: serpentine and nested sorts.
//!
//! A sequential procedure buys variance from the order of the list: units
//! that sit together are selected apart, so the sample spreads over
//! whatever the sort keys encode — implicit stratification. A plain nested
//! sort leaves a seam at every boundary where the last unit of one group
//! meets the first of the next and the two could not be less alike. The
//! serpentine sort removes the seams by reversing direction at each
//! boundary, so the list meanders through the frame and every neighbouring
//! pair of units is actually similar.
//!
//! Categorical keys are ranked by byte order of their UTF-8 encoding —
//! the C locale — deliberately: a sort that follows the machine's locale
//! produces a different sample on a different machine.

use crate::error::Error;

/// One sort key: a column of the frame.
#[derive(Debug, Clone)]
pub enum Key<'a> {
    /// A numeric key, used as is. Must be finite.
    Numeric(&'a [f64]),
    /// A categorical key, ranked by byte order of its values.
    Categorical(&'a [String]),
}

impl Key<'_> {
    fn len(&self) -> usize {
        match self {
            Key::Numeric(v) => v.len(),
            Key::Categorical(v) => v.len(),
        }
    }

    /// The key as ranks: the numeric values themselves, or the 1-based
    /// position of each category in the byte-ordered set of categories.
    fn ranks(&self) -> Result<Vec<f64>, Error> {
        match self {
            Key::Numeric(v) => {
                if let Some(bad) = v.iter().find(|x| !x.is_finite()) {
                    return Err(Error::Input(format!(
                        "sort keys must be finite; found {bad}"
                    )));
                }
                Ok(v.to_vec())
            }
            Key::Categorical(v) => {
                let mut levels: Vec<&str> = v.iter().map(String::as_str).collect();
                levels.sort_unstable();
                levels.dedup();
                Ok(v.iter()
                    .map(|s| (levels.binary_search(&s.as_str()).unwrap() + 1) as f64)
                    .collect())
            }
        }
    }
}

fn rank_columns(keys: &[Key]) -> Result<(Vec<Vec<f64>>, usize), Error> {
    if keys.is_empty() {
        return Err(Error::Input("at least one sort key is needed".into()));
    }
    let n = keys[0].len();
    if n == 0 {
        return Err(Error::Input("sort keys must not be empty".into()));
    }
    let mut ranks = Vec::with_capacity(keys.len());
    for key in keys {
        if key.len() != n {
            return Err(Error::Input(format!(
                "sort keys of different lengths: {} and {}",
                n,
                key.len()
            )));
        }
        ranks.push(key.ranks()?);
    }
    Ok((ranks, n))
}

/// Stable ascending order of `0..n` by the given columns, compared in turn.
fn multi_key_order(ranks: &[Vec<f64>], n: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        for col in ranks {
            match col[a].partial_cmp(&col[b]).unwrap() {
                std::cmp::Ordering::Equal => continue,
                other => return other,
            }
        }
        std::cmp::Ordering::Equal
    });
    order
}

/// The nested sort: plain lexicographic order over the keys.
///
/// Returns the 0-based order in which to list the frame.
pub fn nested(keys: &[Key]) -> Result<Vec<usize>, Error> {
    let (ranks, n) = rank_columns(keys)?;
    Ok(multi_key_order(&ranks, n))
}

/// The hierarchical serpentine sort.
///
/// The first key orders the list; each further key runs forward through
/// the first group the keys above it define, backward through the second,
/// and so on, so the list never jumps. With a single key the serpentine
/// and nested sorts coincide. Returns the 0-based order in which to list
/// the frame.
pub fn serpentine(keys: &[Key]) -> Result<Vec<usize>, Error> {
    let (ranks, n) = rank_columns(keys)?;
    if ranks.len() == 1 {
        return Ok(multi_key_order(&ranks, n));
    }

    let mut order = multi_key_order(&ranks[..1], n);
    for j in 1..ranks.len() {
        // A group boundary is any change in the keys above this one; the
        // direction alternates with the running count of boundaries.
        let mut changes = vec![false; n];
        changes[0] = true;
        for col in &ranks[..j] {
            for p in 1..n {
                changes[p] = changes[p] || col[order[p]] != col[order[p - 1]];
            }
        }
        let mut level = vec![0_i64; n];
        let mut running = 0_i64;
        for p in 0..n {
            running += i64::from(changes[p]);
            level[p] = running;
        }

        let mut positions: Vec<usize> = (0..n).collect();
        positions.sort_by(|&a, &b| {
            level[a].cmp(&level[b]).then_with(|| {
                let direction = |p: usize| if level[p] % 2 == 1 { 1.0 } else { -1.0 };
                let ka = direction(a) * ranks[j][order[a]];
                let kb = direction(b) * ranks[j][order[b]];
                ka.partial_cmp(&kb).unwrap()
            })
        });
        order = positions.iter().map(|&p| order[p]).collect();
    }
    Ok(order)
}
