use std::{
    cmp::Reverse,
    collections::HashMap,
    error::Error,
    fs,
    io::{self, Read},
    path::{Path, PathBuf},
    process,
    time::SystemTime,
};

use mlx_lm::{Cache, CacheCheckpoint, Model, TokenId};
use sha2::{Digest, Sha256};

/// Raised when files an older server wrote must not be read.
const FORMAT: u32 = 1;
const STATE: &str = "-state";

struct Saved {
    checksum: u32,
    bytes: u64,
    used: SystemTime,
    /// How many tokens the file's state ends after.
    end: usize,
}

/// Conversations' K/V and checkpoints as files in one directory, where a later process
/// with the same model finds them. A K/V file holds one block of tokens, or what a
/// checkpoint has of its last block, and is named by a hash of the name before it and its
/// own tokens, and by where it ends. A checkpoint file takes the name of its last K/V
/// file. Every file name ends in a CRC-32 of the file.
pub(crate) struct Disk {
    directory: PathBuf,
    budget: u64,
    block: usize,
    /// By name, without the checksum.
    files: HashMap<String, Saved>,
}

fn hex(hash: Sha256) -> String {
    let bytes = hash.finalize();
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}
fn link(previous: &str, tokens: &[TokenId], end: usize) -> String {
    let mut hash = Sha256::new();
    hash.update(previous);
    for &token in tokens {
        hash.update(u32::from(token).to_le_bytes());
    }
    format!("{}-{end}", hex(hash))
}
fn checksum(mut file: &fs::File) -> io::Result<u32> {
    let mut hash = crc32fast::Hasher::new();
    let mut buffer = vec![0; 1 << 20];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            return Ok(hash.finalize());
        }
        hash.update(&buffer[..read]);
    }
}
fn file_name(name: &str, checksum: u32) -> String {
    format!("{name}.{checksum:08x}.safetensors")
}

impl Disk {
    /// Opens the directory under `root` for this server, the block size, and the model's
    /// name, `config.json` and weight files' paths, sizes and modification times. Drops
    /// what an interrupted write left there and what is over the budget.
    pub fn open(root: &Path, model: &Path, budget: u64, block: usize) -> io::Result<Self> {
        let mut shards = Vec::new();
        for entry in fs::read_dir(model)? {
            let path = entry?.path();
            if path.extension().is_some_and(|it| it == "safetensors") {
                let metadata = fs::metadata(&path)?;
                shards.push((path, metadata.len(), metadata.modified()?));
            }
        }
        shards.sort();
        let mut hash = Sha256::new();
        let version = env!("CARGO_PKG_VERSION");
        hash.update(format!("{FORMAT} {version} {block} {shards:?}"));
        hash.update(fs::read(model.join("config.json"))?);
        let name = model.file_name().unwrap_or_default().to_string_lossy();
        let directory = root.join(format!("{name}-{}", hex(hash)));
        fs::create_dir_all(&directory)?;

        let mut disk = Self {
            directory,
            budget,
            block,
            files: HashMap::new(),
        };
        let now = SystemTime::now();
        for entry in fs::read_dir(&disk.directory)? {
            let entry = entry?;
            let file = entry.file_name();
            let parts = file.to_str().and_then(|file| {
                let (name, checksum) = file.strip_suffix(".safetensors")?.rsplit_once('.')?;
                let checksum = u32::from_str_radix(checksum, 16).ok()?;
                let (_, end) = name.strip_suffix(STATE).unwrap_or(name).split_once('-')?;
                (file == file_name(name, checksum)).then_some((name, checksum, end.parse().ok()?))
            });
            let Some((name, checksum, end)) = parts else {
                let _ = fs::remove_file(entry.path());
                continue;
            };
            let metadata = entry.metadata()?;
            let saved = Saved {
                checksum,
                bytes: metadata.len(),
                // A file dated after now would never be the one used longest ago.
                used: metadata.modified()?.min(now),
                end,
            };
            // Two processes that shared the directory can each have written the name.
            disk.remove(name);
            disk.files.insert(name.to_owned(), saved);
        }
        disk.evict(0);
        Ok(disk)
    }
    fn path(&self, name: &str, checksum: u32) -> PathBuf {
        self.directory.join(file_name(name, checksum))
    }
    fn remove(&mut self, name: &str) {
        if let Some(saved) = self.files.remove(name) {
            let _ = fs::remove_file(self.path(name, saved.checksum));
        }
    }
    /// Marks the file as used at `time`, which a later process reads from its
    /// modification time.
    fn touch(&mut self, name: &str, time: SystemTime) {
        let Some(saved) = self.files.get_mut(name).filter(|saved| saved.used < time) else {
            return;
        };
        saved.used = time;
        let path = self.path(name, self.files[name].checksum);
        let file = fs::File::options().append(true).open(path);
        let _ = file.and_then(|file| file.set_modified(time));
    }
    /// The names of the files of the state after `tokens`, the checkpoint last.
    fn names(&self, tokens: &[TokenId]) -> Vec<String> {
        let mut names: Vec<String> = Vec::new();
        for (index, block) in tokens.chunks(self.block).enumerate() {
            let previous = names.last().map_or("", String::as_str);
            let end = tokens.len().min((index + 1).saturating_mul(self.block));
            names.push(link(previous, block, end));
        }
        names.extend(names.last().map(|last| format!("{last}{STATE}")));
        names
    }
    /// Drops files until `room` more bytes fit the budget. The state used longest ago
    /// loses its checkpoint and then its K/V from the end, so that what stays is a shorter
    /// state.
    fn evict(&mut self, room: u64) {
        let mut total = room + self.files.values().map(|saved| saved.bytes).sum::<u64>();
        while total > self.budget {
            let oldest = self.files.iter().min_by_key(|(name, saved)| {
                (saved.used, Reverse(saved.end), !name.ends_with(STATE))
            });
            let Some((oldest, saved)) = oldest else {
                return;
            };
            total -= saved.bytes;
            self.remove(&oldest.clone());
        }
    }

    /// Marks the files of the states at `checkpoints` as used at `since`, when the request
    /// that used them ended, and writes one that is missing. False when none is, or when
    /// its state does not fit the budget beside the files used at `since`.
    pub fn save(
        &mut self,
        cache: &Cache,
        checkpoints: &[&CacheCheckpoint],
        since: SystemTime,
    ) -> bool {
        let written = self.write(cache, checkpoints, since);
        written.unwrap_or_else(|error| {
            eprintln!("mlx-lm-server: {error}");
            false
        })
    }
    fn write(
        &mut self,
        cache: &Cache,
        checkpoints: &[&CacheCheckpoint],
        since: SystemTime,
    ) -> Result<bool, Box<dyn Error>> {
        let mut missing = None;
        for checkpoint in checkpoints.iter().rev() {
            let names = self.names(checkpoint.tokens());
            for name in &names {
                self.touch(name, since);
            }
            let lacks = |name: &String| !self.files.contains_key(name);
            if let Some(index) = names.iter().position(lacks) {
                missing = Some((checkpoint, names, index));
            }
        }
        let Some((checkpoint, mut names, index)) = missing else {
            return Ok(false);
        };
        let name = names.swap_remove(index);
        let tokens = checkpoint.tokens().len();
        let end = tokens.min((index + 1).saturating_mul(self.block));
        let positions = (!name.ends_with(STATE)).then(|| index * self.block..end);
        let temporary = format!("tmp-{}.safetensors", process::id());
        let temporary = self.directory.join(temporary);
        match &positions {
            Some(positions) => cache.save_positions(positions.clone(), &temporary)?,
            None => checkpoint.save(&temporary)?,
        }
        let file = fs::File::open(&temporary)?;
        let (checksum, bytes) = (checksum(&file)?, file.metadata()?.len());
        // A state that will not fit must not cost older states their files, so the rest of
        // it is judged from this file before any is dropped. The files used at `since`
        // are the newest and stay.
        let rest = match positions {
            Some(positions) => {
                let later = (tokens - positions.end) as u64;
                bytes * later / positions.len() as u64 + checkpoint.nbytes()? as u64
            }
            None => 0,
        };
        let held = self.files.values().filter(|saved| saved.used >= since);
        let held = held.map(|saved| saved.bytes).sum::<u64>();
        if bytes + rest + held > self.budget {
            fs::remove_file(&temporary)?;
            return Ok(false);
        }
        self.evict(bytes);
        file.set_modified(since)?;
        file.sync_all()?;
        fs::rename(&temporary, self.path(&name, checksum))?;
        let saved = Saved {
            checksum,
            bytes,
            used: since,
            end,
        };
        self.files.insert(name, saved);
        Ok(true)
    }

    /// The longest saved state that `tokens` continue and that holds more than `held` of
    /// them, as a cache at its checkpoint. A file that is not what its name says, or that
    /// the library refuses, is removed, and the next longest state is tried.
    pub fn restore(
        &mut self,
        model: &Model,
        tokens: &[TokenId],
        held: usize,
    ) -> Option<(Cache, CacheCheckpoint)> {
        // Generation needs at least one token to prefill.
        let tokens = &tokens[..tokens.len().saturating_sub(1)];
        let states = self.files.iter();
        let states = states.filter(|(name, saved)| {
            name.ends_with(STATE) && held < saved.end && saved.end <= tokens.len()
        });
        let mut ends: Vec<usize> = states.map(|(_, saved)| saved.end).collect();
        ends.sort();
        ends.dedup();
        while let Some(end) = ends.pop() {
            if let Some(loaded) = self.load(model, &tokens[..end]) {
                return Some(loaded);
            }
        }
        None
    }
    fn load(&mut self, model: &Model, tokens: &[TokenId]) -> Option<(Cache, CacheCheckpoint)> {
        let names = self.names(tokens);
        if !names.iter().all(|name| self.files.contains_key(name)) {
            return None;
        }
        let now = SystemTime::now();
        let mut paths = Vec::new();
        for name in &names {
            self.touch(name, now);
            let path = self.path(name, self.files[name].checksum);
            let found = fs::File::open(&path).and_then(|file| checksum(&file));
            if found.ok() != Some(self.files[name].checksum) {
                self.remove(name);
                return None;
            }
            paths.push(path);
        }
        let (state, blocks) = paths.split_last()?;
        let checkpoint = model.load_checkpoint(state).ok();
        // A checkpoint of other tokens under this name would fail every request for them.
        let checkpoint = checkpoint.filter(|checkpoint| checkpoint.tokens() == tokens);
        let Some(checkpoint) = checkpoint else {
            self.remove(names.last()?);
            return None;
        };
        let Ok(cache) = model.load_cache(&checkpoint, blocks) else {
            for name in &names {
                self.remove(name);
            }
            return None;
        };
        Some((cache, checkpoint))
    }
}
