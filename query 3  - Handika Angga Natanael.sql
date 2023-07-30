select sum (qty) as total_qty, store.storename
from "transaction" t  
left outer join store
on t.storeid = store.storeid
group by store.storename
order by total_qty desc
limit 1

