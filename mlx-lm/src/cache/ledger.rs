use super::{invalid, CacheError};
use crate::TokenId;
use std::rc::Rc;

#[derive(Clone, Default)]
pub(super) struct TokenLedger {
    pub(super) ids: Rc<Vec<TokenId>>,
}

pub(super) struct PendingTokens {
    ids: Vec<TokenId>,
    replacement: Option<Rc<Vec<TokenId>>>,
}

impl TokenLedger {
    pub(super) fn tokens(&self) -> &[TokenId] {
        &self.ids
    }

    pub(super) fn reserve(&mut self, capacity: usize) -> Result<(), CacheError> {
        let additional = capacity.saturating_sub(self.ids.len());
        Rc::make_mut(&mut self.ids)
            .try_reserve(additional)
            .map_err(|_| invalid("cannot reserve token ledger"))
    }

    pub(super) fn prepare(&mut self, ids: &[TokenId]) -> Result<PendingTokens, CacheError> {
        let mut staged = Vec::new();
        staged
            .try_reserve(ids.len())
            .map_err(|_| invalid("cannot stage token IDs"))?;
        staged.extend_from_slice(ids);
        let replacement = if let Some(current) = Rc::get_mut(&mut self.ids) {
            current
                .try_reserve(ids.len())
                .map_err(|_| invalid("cannot reserve token ledger"))?;
            None
        } else {
            let length = self
                .ids
                .len()
                .checked_add(ids.len())
                .ok_or_else(|| invalid("token ledger overflow"))?;
            let mut copy = Vec::new();
            copy.try_reserve(self.ids.capacity().max(length))
                .map_err(|_| invalid("cannot copy token ledger"))?;
            copy.extend_from_slice(&self.ids);
            Some(Rc::new(copy))
        };
        Ok(PendingTokens {
            ids: staged,
            replacement,
        })
    }

    pub(super) fn commit(&mut self, pending: PendingTokens) {
        if let Some(replacement) = pending.replacement {
            self.ids = replacement;
        }
        // The step's exclusive cache borrow prevents new snapshot owners after reservation.
        Rc::make_mut(&mut self.ids).extend_from_slice(&pending.ids);
    }
}
